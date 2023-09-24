import os
import reverb
import tempfile

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.sac import sac_agent
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import PolicySaver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

import shutil

from cigre_pv_wind_PPEnv import CigrePPEnv


#This file is mainly copied from Tensorflow SAC Minitaur example
#https://www.tensorflow.org/agents/tutorials/7_SAC_minitaur_tutorial

tempdir = tempfile.gettempdir()

eval_save_dir = os.path.join(tempdir, 'eval')

if os.path.exists(eval_save_dir):
    shutil.rmtree(eval_save_dir)


##########################################################
# Hyperparamertes
##########################################################


num_iterations = 1500000 # @param {type:"integer"} number of train steps

initial_collect_steps = 50000 # @param {type:"integer"} number of random steps in the beginning
replay_buffer_capacity = 1000000 # @param {type:"integer"} 
#smaller replay buffer capacity than num_iterations can lead to big mistakes in later part of the training if there are no more bad experiences in buffer anymore. In this training it is partly avoided by deleting old replays uniformly and not by fifo

batch_size = 1000 # @param {type:"integer"} #number of fetched steps from replay buffer per training dataset. Can be varied quiet a bit but smaller batch_sizes lead to longer training time


critic_learning_rate = 3e-4 # @param {type:"number"} learning rate for critic NN
actor_learning_rate = 3e-4 # @param {type:"number"} learning rate for actor NN
alpha_learning_rate = 3e-4 # @param {type:"number"} learning rate for alpha factor which regulates entropy which is the regulating factor for exploration/exploitation tradeoff in SAC
#more on SAC here https://spinningup.openai.com/en/latest/algorithms/sac.html

#haven't changed these factors from the example
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}

reward_scale_factor = 1.0 # @param {type:"number"} at some point I set the factor to two but since reward is also scaled in env small changes should not matter to much

#fully connected layer describtion of NN
actor_fc_layer_params = (100, 100)
critic_joint_fc_layer_params = (100, 100)

num_eval_episodes = 1 # @param {type:"integer"}
train_episodes_per_eval_episode = 2 # @param {type:"integer"}


##########################################################
# Environment
##########################################################
#env = CigrePPEnv(False, "", [1], 0)
#env.reset()

#print('Observation Spec:')
#print(env.time_step_spec().observation)
#print('Action Spec:')
#print(env.action_spec())

#utils.validate_py_environment(env, episodes=2)



collect_env = CigrePPEnv(False, "collect_env_log.db",[1,3,5,7,9,11], 1, "collect")
eval_env = CigrePPEnv(True, "eval_env_log.db", [2,4,6,8,10,12], 0, "evaluation")



##########################################################
# Distribution Strategy
##########################################################
#strategy describes the hardware usesage which depends on the training platform
use_gpu = False

strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)



##########################################################
# Agent
##########################################################
observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(collect_env))


with strategy.scope():
  critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')


#used ActorNetwork instead of ActorDistributionNetwork to get deterministic results
#set activation layer to None if negative value should be trained although input/output normalization is advised
with strategy.scope():
    actor_net = actor_network.ActorNetwork(
      observation_spec,
      action_spec,
      fc_layer_params=actor_fc_layer_params,
      activation_fn=tf.keras.activations.relu)


with strategy.scope():
  train_step = train_utils.create_train_step()

  tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step)

  tf_agent.initialize()



##########################################################
# Replay Buffer
##########################################################
#not really changed these variables from example, test trainings of different values 2-100 seemed not to make a big difference
rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)

#mainly as in example, just changed remover to Uniform to avoid bad results if the buffer starts deleting the first experiences
table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Uniform(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([table])

reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server)


#preftch(tf.data.AUTOTUNE) did not seem to make a runtime improvement
dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(1)
experience_dataset_fn = lambda: dataset



##########################################################
# Policies
##########################################################
#eager polcies just accelerate the training
tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_eval_policy, use_tf_function=True)

tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_collect_policy, use_tf_function=True)


random_policy = random_py_policy.RandomPyPolicy(
  collect_env.time_step_spec(), collect_env.action_spec())



##########################################################
# Actors
##########################################################
rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  reverb_replay.py_client,
  table_name,
  sequence_length=2,
  stride_length=1)


initial_collect_actor = actor.Actor(
  collect_env,
  random_policy,
  train_step,
  steps_per_run=initial_collect_steps,
  observers=[rb_observer])

print(f"Initial Collect Actor")
initial_collect_actor.run()
print(f"Initial Collect Actor finished")

#smaller collect metric, no summary_dir leading to no collect metric both did not make a significant runtime improvement
env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
  collect_env,
  collect_policy,
  train_step,
  steps_per_run=1,
  metrics=actor.collect_metrics(10),
  summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
  observers=[rb_observer, env_step_metric])


eval_actor = actor.Actor(
  eval_env,
  eval_policy,
  train_step,
  episodes_per_run=num_eval_episodes,
  metrics=actor.eval_metrics(num_eval_episodes),
  summary_dir=eval_save_dir,
)

saver = PolicySaver(tf_agent.policy)


##########################################################
# Learners
##########################################################

agent_learner = learner.Learner(
  tempdir,
  train_step,
  tf_agent,
  experience_dataset_fn,
  strategy=strategy)


##########################################################
# Metrics and Evaluation
##########################################################
def get_eval_metrics():
  eval_actor.run()
  results = {}
  for metric in eval_actor.metrics:
    results[metric.name] = metric.result()
  return results

#metrics = get_eval_metrics()

#logs AverageReturn as sum of rewards per Episode
def log_eval_metrics(step, metrics):
  eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
  print('step = {0}: {1}'.format(step, eval_results))
  with open("training_log.txt", "a") as logfile:
    logfile.write('step = {0}: {1}\n'.format(step, eval_results))

#log_eval_metrics(0, metrics)


##########################################################
# Training
##########################################################
#try:
#  %%time
#except:
#  pass

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
max_return = avg_return
print(f"Initial max return set to {max_return}")


print(f"Training")

train_episode_counter = 0

average_total_loss = 0
average_actor_loss = 0
average_critic_loss = 0
average_alpha_loss = 0
episode_step_counter = 0

for _ in range(num_iterations):
  # Training, collect one step, train one step from replay buffer
  collect_actor.run()
  loss_info = agent_learner.run(iterations=1)
  episode_step_counter += 1
  
  average_total_loss += loss_info.loss.numpy()
  average_actor_loss += loss_info.extra.actor_loss.numpy()
  average_critic_loss += loss_info.extra.critic_loss.numpy()
  average_alpha_loss += loss_info.extra.alpha_loss.numpy()
  
  #print(f"actor_loss: {loss_info.extra.actor_loss.numpy()} ; average_critic_loss:{loss_info.extra.critic_loss.numpy()} ; alpha_loss:{loss_info.extra.alpha_loss.numpy()}")
  
  #print(loss_info)

  # Evaluating.
  step = agent_learner.train_step_numpy

  if collect_env.is_episode_finished():
    train_episode_counter += 1
    
    average_total_loss /= episode_step_counter
    average_actor_loss /= episode_step_counter
    average_critic_loss /= episode_step_counter
    average_alpha_loss /= episode_step_counter
    
    
    print('step = {0}: loss = {1}'.format(step, average_total_loss))
    print('average_actor_loss = {0} : average_critic_loss = {1} : average_alpha_loss={2}\n'.format(average_actor_loss, average_critic_loss, average_alpha_loss))
    with open("training_log.txt", "a") as logfile:
      logfile.write('step = {0}: loss = {1}\n'.format(step, average_total_loss))
      logfile.write('average_actor_loss = {0} : average_critic_loss = {1} : average_alpha_loss={2}\n'.format(average_actor_loss, average_critic_loss, average_alpha_loss))
  
    average_total_loss = 0
    average_actor_loss = 0
    average_critic_loss = 0
    average_alpha_loss = 0
    episode_step_counter = 0
    
    if train_episode_counter == train_episodes_per_eval_episode:
      print(f"eval episode at step {step}")
      metrics = get_eval_metrics()
      log_eval_metrics(step, metrics)
      train_episode_counter = 0
    
      #export best actor network, prevents overfitting
      if (metrics["AverageReturn"] > max_return):
        print(f"New best eval, saving policy")
        max_return = metrics["AverageReturn"]
        saver.save('actor_policy')
        #saver.save_checkpoint('actor_policy_checkpoint')


rb_observer.close()
reverb_server.stop()

shutil.rmtree(eval_save_dir)
shutil.rmtree(tempdir)
