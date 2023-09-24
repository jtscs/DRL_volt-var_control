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
from simEnv_interface import simEnvInterface

import custom_actor


#This file is mainly copied from Tensorflow SAC Minitaur example
#https://www.tensorflow.org/agents/tutorials/7_SAC_minitaur_tutorial


global_collect_env = CigrePPEnv(False, "collect_env_log.db", [1, 3, 5, 7, 9, 11], 1, "collect")
global_eval_env = CigrePPEnv(True, "eval_env_log.db", [2, 4, 6, 8, 10, 12], 0, "evaluation")


class SAC_Learner():


    ##########################################################
    # Hyperparamertes
    ##########################################################

    replay_buffer_capacity = 150000  # @param {type:"integer"}
    # smaller replay buffer capacity than num_iterations can lead to big mistakes in later part of the training if there are no more bad experiences in buffer anymore. In this training it is partly avoided by deleting old replays uniformly and not by fifo

    batch_size = 1000  # @param {type:"integer"} #number of fetched steps from replay buffer per training dataset. Can be varied quiet a bit but smaller batch_sizes lead to longer training time

    critic_learning_rate = 3e-4  # @param {type:"number"} learning rate for critic NN
    actor_learning_rate = 3e-4  # @param {type:"number"} learning rate for actor NN
    alpha_learning_rate = 3e-4  # @param {type:"number"} learning rate for alpha factor which regulates entropy which is the regulating factor for exploration/exploitation tradeoff in SAC
    # more on SAC here https://spinningup.openai.com/en/latest/algorithms/sac.html

    # haven't changed these factors from the example
    target_update_tau = 0.005  # @param {type:"number"}
    target_update_period = 1  # @param {type:"number"}
    gamma = 0.99  # @param {type:"number"}

    reward_scale_factor = 1.0  # @param {type:"number"} at some point I set the factor to two but since reward is also scaled in env small changes should not matter to much

    # fully connected layer describtion of NN
    actor_fc_layer_params = (100, 100)
    critic_joint_fc_layer_params = (100, 100)

    num_eval_episodes = 1  # @param {type:"integer"}


    max_average_episode_return = -999999999


    def __init__(self, name, id):

        self.name = name
        self.id = id

        tempdir = tempfile.gettempdir()

        eval_save_dir = os.path.join(tempdir, 'eval_'+name)
        
        #print(f"eval_save_dir {eval_save_dir}")

        if os.path.exists(eval_save_dir):
            shutil.rmtree(eval_save_dir)


        ##########################################################
        # Environment
        ##########################################################

        collect_env = simEnvInterface(self.name, self.id, global_collect_env)
        eval_env = simEnvInterface(self.name, self.id, global_eval_env)


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
              joint_fc_layer_params=self.critic_joint_fc_layer_params,
              kernel_initializer='glorot_uniform',
              last_kernel_initializer='glorot_uniform')


        #used ActorNetwork instead of ActorDistributionNetwork to get deterministic results
        #set activation layer to None if negative value should be trained although input/output normalization is advised
        with strategy.scope():
            actor_net = actor_network.ActorNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=self.actor_fc_layer_params,
            activation_fn=tf.keras.activations.relu)


        with strategy.scope():
            train_step = train_utils.create_train_step()

            tf_agent = sac_agent.SacAgent(
                time_step_spec,
                action_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.keras.optimizers.Adam(learning_rate=self.actor_learning_rate),
                critic_optimizer=tf.keras.optimizers.Adam(learning_rate=self.critic_learning_rate),
                alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha_learning_rate),
                target_update_tau=self.target_update_tau,
                target_update_period=self.target_update_period,
                td_errors_loss_fn=tf.math.squared_difference,
                gamma=self.gamma,
                reward_scale_factor=self.reward_scale_factor,
                train_step_counter=train_step)

            tf_agent.initialize()



        ##########################################################
        # Replay Buffer
        ##########################################################
        #not really changed these variables from example, test trainings of different values 2-100 seemed not to make a big difference
        rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)

        #mainly as in example, just changed remover to Uniform to avoid bad results if the buffer starts deleting the first experiences
        self.table_name = 'uniform_table'
        self.table = reverb.Table(
            self.table_name,
            max_size=self.replay_buffer_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Uniform(),
            rate_limiter=reverb.rate_limiters.MinSize(1))

        self.reverb_server = reverb.Server([self.table])

        self.reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
            tf_agent.collect_data_spec,
            sequence_length=2,
            table_name=self.table_name,
            local_server=self.reverb_server)



        self.dataset = self.reverb_replay.as_dataset(
              sample_batch_size=self.batch_size, num_steps=2).prefetch(1)
        self.experience_dataset_fn = lambda: self.dataset



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
        self.rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
          self.reverb_replay.py_client,
          self.table_name,
          sequence_length=2,
          stride_length=1)


        self.initial_collect_actor = custom_actor.CustomActor(
          collect_env,
          random_policy,
          train_step,
          steps_per_run=1,
          observers=[self.rb_observer])


        self.env_step_metric = py_metrics.EnvironmentSteps()

        self.collect_actor = custom_actor.CustomActor(
          collect_env,
          collect_policy,
          train_step,
          steps_per_run=1,
          metrics=actor.collect_metrics(10),
          summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
          observers=[self.rb_observer, self.env_step_metric])


        self.eval_actor = custom_actor.CustomActor(
          eval_env,
          eval_policy,
          train_step,
          steps_per_run=1,
          metrics=actor.eval_metrics(self.num_eval_episodes),
          summary_dir=eval_save_dir,
        )

        self.saver = PolicySaver(tf_agent.policy)


        ##########################################################
        # Learners
        ##########################################################

        self.agent_learner = learner.Learner(
          tempdir,
          train_step,
          tf_agent,
          self.experience_dataset_fn,
          strategy=strategy)



        # Reset the train step
        tf_agent.train_step_counter.assign(0)


    ##########################################################
    # Metrics and Evaluation
    ##########################################################
    def start_eval_actor(self):
        return self.eval_actor.run()

    def finish_eval_actor(self):
        self.eval_actor.get_run_results()

    def get_eval_metrics(self):
        results = {}
        for metric in self.eval_actor.metrics:
          results[metric.name] = metric.result()

        if self.max_average_episode_return < results["AverageReturn"]:
            self.max_average_episode_return = results["AverageReturn"]
            print(f"New best {self.name} eval, saving policy")
            self.saver.save('actor_policy_'+self.name)

        return results

    #metrics = get_eval_metrics()

    #logs AverageReturn as sum of rewards per Episode
    def log_eval_metrics(self, step, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('{0} step = {1}: {2}'.format(self.name, step, eval_results))
        #with open("training_log.txt", "a") as logfile:
        #  logfile.write('step = {0}: {1}\n'.format(step, eval_results))

    #log_eval_metrics(0, metrics)


    def start_collect_actor(self):
        #print(f"{self.name} start collect actor step")
        return self.collect_actor.run()


    def finish_collect_actor(self):
        #print(f"{self.name} finish collect actor step")
        return self.collect_actor.get_run_results()



    def start_initial_collect_actor(self):
        #print(f"{self.name} start initial collect actor step")
        return self.initial_collect_actor.run()

    def finish_initial_collect_actor(self):
        #print(f"{self.name} finish initial collect actor step")
        return self.initial_collect_actor.get_run_results()


    def run_learner(self):
        #print(f"{self.name} running learner")
        loss_info = self.agent_learner.run(iterations=1)

        # Evaluating.
        step = self.agent_learner.train_step_numpy

        return loss_info, step


    def close(self):
      self.rb_observer.close()
      self.reverb_server.stop()
      





def run_eval_episode(RES_SAC_Learner_List):
    while (not global_eval_env.is_episode_finished()):
        RES_action_list = []
        for SAC_Learner in RES_SAC_Learner_List : RES_action_list.append(SAC_Learner.start_eval_actor())

        action_parameters = [RES_action_list[0][0], RES_action_list[1][0], RES_action_list[2][0], RES_action_list[3][0], RES_action_list[4][0], RES_action_list[5][0], RES_action_list[6][0], RES_action_list[7][0], RES_action_list[8][0]]
        
        #print(f"eval action parameters {action_parameters}")

        global_eval_env.step(action_parameters)

        for SAC_Learner in RES_SAC_Learner_List : SAC_Learner.finish_eval_actor()


    RES_eval_metrics_list = []
    for SAC_Learner in RES_SAC_Learner_List : RES_eval_metrics_list.append(SAC_Learner.get_eval_metrics())
    
    for index, SAC_Learner in enumerate(RES_SAC_Learner_List) : SAC_Learner.log_eval_metrics(0, RES_eval_metrics_list[index])

    global_eval_env.reset()


if __name__ == "__main__":

    num_iterations = 1500000  # @param {type:"integer"} number of train steps
    initial_collect_steps = 50000  # @param {type:"integer"} number of random steps in the beginning
    train_episodes_per_eval_episode = 2  # @param {type:"integer"}

    ##########################################################
    # Training
    ##########################################################

    RES_SAC_Learner_List = [SAC_Learner("RES1",0), SAC_Learner("RES2",1), SAC_Learner("RES3",2), SAC_Learner("RES4",3), SAC_Learner("RES5",4), SAC_Learner("RES6",5), SAC_Learner("RES7",6), SAC_Learner("RES8",7), SAC_Learner("RES9",8)]

    global_eval_env.reset()
    global_collect_env.reset()

    print(f"Start Initial Collect Actors")
    for i in range(initial_collect_steps):
    
        RES_action_list = []
        for SAC_Learner in RES_SAC_Learner_List : RES_action_list.append(SAC_Learner.start_initial_collect_actor())
        
        action_parameters = [RES_action_list[0][0], RES_action_list[1][0], RES_action_list[2][0], RES_action_list[3][0], RES_action_list[4][0],
                             RES_action_list[5][0], RES_action_list[6][0], RES_action_list[7][0], RES_action_list[8][0]]
                             
        #print(f"collect action parameters {action_parameters}")
        
        global_collect_env._step(action_parameters)
        
        for SAC_Learner in RES_SAC_Learner_List : SAC_Learner.finish_initial_collect_actor()


    global_collect_env.reset()

    print(f"Initial Collect Actors finished")




    # Evaluate the agent's policy once before training.
    run_eval_episode(RES_SAC_Learner_List)
    




    print(f"Training")

    train_episode_counter = 0

    for _ in range(num_iterations):

        # Training, collect one step, train one step from replay buffer
        RES_action_list = []
        for SAC_Learner in RES_SAC_Learner_List : RES_action_list.append(SAC_Learner.start_collect_actor())
        
        action_parameters = [RES_action_list[0][0], RES_action_list[1][0], RES_action_list[2][0], RES_action_list[3][0], RES_action_list[4][0],
                            RES_action_list[5][0], RES_action_list[6][0], RES_action_list[7][0], RES_action_list[8][0]]

        global_collect_env._step(action_parameters)
        
        for SAC_Learner in RES_SAC_Learner_List : SAC_Learner.finish_collect_actor()



        #RES_loss_info_and_step_list = []
        #for SAC_Learner in RES_SAC_Learner_List : RES_loss_info_and_step_list.append(SAC_Learner.run_learner())


        RES1_loss_info, RES1_step = RES_SAC_Learner_List[0].run_learner()
        RES2_loss_info, RES2_step = RES_SAC_Learner_List[1].run_learner()
        RES3_loss_info, RES3_step = RES_SAC_Learner_List[2].run_learner()
        RES4_loss_info, RES4_step = RES_SAC_Learner_List[3].run_learner()
        RES5_loss_info, RES5_step = RES_SAC_Learner_List[4].run_learner()
        RES6_loss_info, RES6_step = RES_SAC_Learner_List[5].run_learner()
        RES7_loss_info, RES7_step = RES_SAC_Learner_List[6].run_learner()
        RES8_loss_info, RES8_step = RES_SAC_Learner_List[7].run_learner()
        RES9_loss_info, RES9_step = RES_SAC_Learner_List[8].run_learner()


        if global_collect_env.is_episode_finished():
            train_episode_counter += 1
            print('RES1 step = {0}: loss = {1}'.format(RES1_step, RES1_loss_info.loss.numpy()))
            print('RES2 step = {0}: loss = {1}'.format(RES2_step, RES2_loss_info.loss.numpy()))
            print('RES3 step = {0}: loss = {1}'.format(RES3_step, RES3_loss_info.loss.numpy()))
            print('RES4 step = {0}: loss = {1}'.format(RES4_step, RES4_loss_info.loss.numpy()))
            print('RES5 step = {0}: loss = {1}'.format(RES5_step, RES5_loss_info.loss.numpy()))
            print('RES6 step = {0}: loss = {1}'.format(RES6_step, RES6_loss_info.loss.numpy()))
            print('RES7 step = {0}: loss = {1}'.format(RES7_step, RES7_loss_info.loss.numpy()))
            print('RES8 step = {0}: loss = {1}'.format(RES8_step, RES8_loss_info.loss.numpy()))
            print('RES9 step = {0}: loss = {1}'.format(RES9_step, RES9_loss_info.loss.numpy()))
            with open("training_log.txt", "a") as logfile:
               logfile.write('RES1 step = {0}: loss = {1}\n'.format(RES1_step, RES1_loss_info.loss.numpy()))
               logfile.write('RES2 step = {0}: loss = {1}\n'.format(RES2_step, RES2_loss_info.loss.numpy()))
               logfile.write('RES3 step = {0}: loss = {1}\n'.format(RES3_step, RES3_loss_info.loss.numpy()))
               logfile.write('RES4 step = {0}: loss = {1}\n'.format(RES4_step, RES4_loss_info.loss.numpy()))
               logfile.write('RES5 step = {0}: loss = {1}\n'.format(RES5_step, RES5_loss_info.loss.numpy()))
               logfile.write('RES6 step = {0}: loss = {1}\n'.format(RES6_step, RES6_loss_info.loss.numpy()))
               logfile.write('RES7 step = {0}: loss = {1}\n'.format(RES7_step, RES7_loss_info.loss.numpy()))
               logfile.write('RES8 step = {0}: loss = {1}\n'.format(RES8_step, RES8_loss_info.loss.numpy()))
               logfile.write('RES9 step = {0}: loss = {1}\n\n'.format(RES9_step, RES9_loss_info.loss.numpy()))

        if train_episode_counter == train_episodes_per_eval_episode:
            print(f"eval episode at step {RES1_step}")
            # Evaluate the agent's policy once before training.
            run_eval_episode(RES_SAC_Learner_List)

            train_episode_counter = 0



    for SAC_Learner in RES_SAC_Learner_List : SAC_Learner.close()

