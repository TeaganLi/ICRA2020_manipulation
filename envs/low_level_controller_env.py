"""
3 Finger Robot Gym Environment.
"""
import os, logging, gym
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client
from gym import spaces

from utils.dynamics_calculator import DynamicCalculator, DynamicCalculatorCube
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from .base_env import ThreeFingerRobot

logger = logging.getLogger(__name__)
ORI_THRESH = 0.1
DIST_THRESH = 0.01
GOAL_THRESH = (DIST_THRESH, DIST_THRESH, ORI_THRESH)
GOAL_RANGE_LOW = (-0.05, -0.05, -np.pi/2)
GOAL_RANGE_HIGH = (0.05, 0.05, np.pi/2)
REW_MIN_CLIP = -2

DELTA_REPOSE = (0.01, 0.01, 0.1)
DELTA_SLIDE = (0.02, 0.02, 0.02)

REPOSE_DUR = 30
SLIDE_DUR = 30
FLIP_DUR = 1200
CAM_DIST = 0.5
RENDER_WIDTH = 1200
RENDER_HEIGHT = 720
DEBUG = False


class Gripper2DPrimitiveEnv(MJCFBaseBulletEnv):
    """
    Base env for low-level torque controller
    """

    def __init__(self, urdf_name, render, measurement_err, hide_target, hide_target_local, init_file,   # task dependent
                 repose_duration=REPOSE_DUR, slide_duration=SLIDE_DUR, flip_duration=FLIP_DUR,
                 delta_repose=DELTA_REPOSE, delta_slide=DELTA_SLIDE, distance_threshold=DIST_THRESH,
                 orientation_threshold=ORI_THRESH, frame_skip=1, time_step=1/60, max_episode_steps=200,
                 cam_dist=CAM_DIST, render_width=RENDER_WIDTH, render_height=RENDER_HEIGHT,
                 ):
        self.robot = ThreeFingerRobot(urdf_name=urdf_name, measurement_err=measurement_err, init_file=init_file,
                                      hide_target=hide_target, hide_target_local=hide_target_local)
        super().__init__(self.robot, render)
        self.repose_duration, self.slide_duration, self.flip_duration = repose_duration, slide_duration, flip_duration
        self.distance_threshold, self.orientation_threshold = distance_threshold, orientation_threshold
        self.frame_skip, self.time_step, self.max_episode_steps = frame_skip, time_step, max_episode_steps
        self._cam_dist, self._render_width, self._render_height, self._cam_yaw, self._cam_pitch = cam_dist, render_width, render_height, 0, -90
        self.delta_repose = np.array([[delta_repose[0], 0, 0], [-delta_repose[0], 0, 0], [0, delta_repose[1], 0],
                                      [0, -delta_repose[1], 0], [0, 0, delta_repose[2]], [0, 0, -delta_repose[2]]])
        self.delta_slide = np.array([[delta_slide[0], 0, 0], [-delta_slide[0], 0, 0], [0, delta_slide[1], 0],
                                     [0, -delta_slide[1], 0], [0, 0, delta_slide[2]], [0, 0, -delta_slide[2]]])
        self.flip_done, self.init_file = False, init_file

    def create_single_player_scene(self, bullet_client):
        # set the gravity in reset function
        return SingleRobotEmptyScene(bullet_client, gravity=0, timestep=self.time_step, frame_skip=self.frame_skip)

    def update_indicator(self, pose, ds):
        """
        Update the pose of object, locations of contact points
        """
        for i in range(3):
            self.robot.target_local_pole_joints[i].set_state(pose[i], 0)
            self.robot.target_contact_point_joints[i].set_state(ds[i], 0)
        return

    def is_success(self, achieved_goal, desired_goal):
        x_dist, y_dist, theta_dist = desired_goal - achieved_goal
        return (np.abs(x_dist) < self.distance_threshold) and \
               (np.abs(y_dist) < self.distance_threshold) and \
               (np.abs(theta_dist) < self.orientation_threshold)

    def is_lose_contact(self):
        """If lose two contacts, the episode is failed"""
        C1 = self.robot._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=1, linkIndexB=8)
        C2 = self.robot._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=3, linkIndexB=8)
        C3 = self.robot._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=5, linkIndexB=8)
        lose_contact_num = np.sum([len(c) == 0 for c in [C1, C2, C3]])
        return lose_contact_num >= 2

    def compute_reward(self, achieved_goal, desired_goal, feasibility):
        if self.reward_type == 'dense':
            reward = 0
        else:
            epi_done = self.is_success(achieved_goal, desired_goal)
            reward = 5 * float(epi_done) - 0.01*(not feasibility) + 1*self.flip_done  # invalid action penalt
            self.flip_done = False
        return reward

    def reset(self):
        """
        Overwrite MJCF reset method for initialize purpose. Hard, medium, easy have different init & goal generation method.
        Hard: init are sampled from f1 and goal is f2 (currently)
        """
        super().reset()
        self._p.setGravity(0, 0, 0)
        # --------------init & goal init --------------------
        if self.init_file is None:
            while 1:
                self.robot.object_pos, self.robot.target_pos, q_init, self.mode = self._sample_pose()
                if self.init_file is not None:
                    break
                for i in range(len(self.robot.object_pos)):
                    self.robot.object_pole_joints[i].set_state(self.robot.object_pos[i], 0)
                    self.robot.target_pole_joints[i].set_state(self.robot.target_pos[i], 0)
                for i in range(len(q_init)):
                    self.robot.manipulator_joints[i].set_state(q_init[i], 0)
                self._p.setJointMotorControlArray(0, jointIndices=[i for i in range(9)],
                                                  controlMode=p.VELOCITY_CONTROL, forces=[0] * 9)
                self._p.stepSimulation()
                _, _, _, xdot = self.robot.get_state()
                if max(abs(xdot)) > 1: continue
                collision_1 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=0, linkIndexB=8)
                collision_2 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=2, linkIndexB=8)
                collision_3 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=4, linkIndexB=8)
                if len(collision_1) > 0 or len(collision_2) > 0 or len(collision_3) > 0: continue
                d1, d2, d3 = self.robot.get_contact_pts(self.controller.compute_forward_kinematics(self.robot.get_state()[0]))
                if None in [d1, d2, d3]: continue
                if np.max(np.abs([d1,d2,d3])) > self.object_length/2. - 0.02: continue
                if None not in [d1,d2,d3] and (np.abs(d1-d2) > 0.05 and np.abs(d2-d3) > 0.05 and np.abs(d1-d3) > 0.05): break    # some init are invalid, re-sample pose
        s = self.robot.calc_state()
        self._p.setGravity(0, self.gravity, 0)
        return s

    def _contact_loc_parser(self, file_name):
        """
        Given a file name (str), parse its contact locations.
        file name is like -0.10,0.04,0.14.npy
        """
        file_name = file_name.split('.npy')
        if len(file_name) < 2: return None  # invalid files like .DS_Store
        file_name = file_name[0]
        contact_loc = [float(di) for di in file_name.split(",")]
        return contact_loc

    def _sample_pose(self):
        """
        Given a file_path, sample a valid object pose under a random contact configuration
        """
        def sample_a_pose(file_path, mode, file_name=None):
            file_list = os.listdir(file_path)
            while 1:
                file_name = '0.0,-0.15,0.15.npy'  # self.np_random.choice(file_list) if file_name is None else file_name
                ds = self._contact_loc_parser(file_name)
                if ds is None: continue
                # init pose
                d1, d2, d3 = ds
                object_poses = np.load(os.path.join(file_path, file_name))
                for _ in range(5):
                    object_pos = object_poses[self.np_random.randint(len(object_poses))]
                    object_pos += np.array([self.np_random.uniform(-0.005, 0.005),
                                            self.np_random.uniform(-0.005, 0.005),
                                            self.np_random.uniform(-0.025, 0.025)])
                    q = self.controller.compute_IK(object_pos, d1, d2, d3, mode=mode)
                    if self._check_link_object_collision(q, object_pos[2], mode=mode):
                        return object_pos, q
            return

        rng_set = {'easy': [[0, 0], [1, 1], [2, 2]],
                   'medium': [[0, 0], [1, 1], [2, 2]],
                   'hard': [[0, 1], [0, 2], [0, 0], [1, 1], [2, 2]]}[self.level]  # Hard goal
        rng_id = self.np_random.randint(len(rng_set))
        init_id, goal_id = rng_set[rng_id]
        if self.level == 'hard':
            self.need_flip_f2 = True if rng_id == 0 else False
            self.need_flip_f3 = True if rng_id == 1 else False
        file_paths, modes = [self.f1_file_path, self.f2_file_path, self.f3_file_path], ['f1', 'f2', 'f3']
        init_file_path, init_mode = file_paths[init_id], modes[init_id]
        goal_file_path, goal_mode = file_paths[goal_id], modes[goal_id]

        if self.level == 'easy':
            file_name = self.np_random.choice(os.listdir(init_file_path))
            init_object_pos, init_q = sample_a_pose(init_file_path, init_mode, file_name)
            goal_object_pos, _ = sample_a_pose(goal_file_path, goal_mode, file_name)
        else:
            init_object_pos, init_q = sample_a_pose(init_file_path, init_mode)
            goal_object_pos, _ = sample_a_pose(goal_file_path, goal_mode)
        return init_object_pos, goal_object_pos, init_q, init_mode

    def _check_link_object_collision(self, q, theta, mode):
        """
        check whether finger link collide with
        """
        if q is None: return False
        if mode == 'f1':
            if q[0] + q[1] >= np.pi + theta: return False  # f2 link 2
            if q[2] + q[3] <= theta:         return False  # f2 link 2
            if q[4] + q[5] >= np.pi + theta: return False  # f3 link 2
        elif mode == "f2":
            if q[0] + q[1] >= np.pi + theta: return False
            if q[2] + q[3] <= theta:         return False
            if q[4] + q[5] <= np.pi + theta: return False
        elif mode == "f3":
            if q[0] + q[1] >= np.pi + theta: return False
            if q[2] + q[3] >= theta:         return False
            if q[4] + q[5] >= np.pi + theta: return False
        return True

    def set_bullet_file(self, file_path=None):
        """Set bullet file so that restore from saved states
            If None, randomly init & goal pose
        """
        self.init_file = file_path
        self.robot.init_file = file_path
        return


class Gripper2DPrimitiveDiscreteReposeEnv(Gripper2DPrimitiveEnv):
    """
    Three finger env -- Easy,
    6 discrete actions, 18 observation dim
    Support 3 shapes, rectangle with different length and width
    """
    def __init__(self, urdf_name="three_fingers_pole_indicator.urdf", render=False, measurement_err=False,
                 hide_target=False, hide_target_local=False, init_file=None, reward_type='sparse', seed=0,
                 is_feasible_check=True, controller_object_length=0.5, controller_object_width=0.02, gravity=0.0, is_render=False,
                 cam_dist=CAM_DIST,
                 ):
        super().__init__(urdf_name=urdf_name, render=render, measurement_err=measurement_err, hide_target=hide_target,
                         hide_target_local=hide_target_local, init_file=init_file,
                         cam_dist=cam_dist)
        self.controller = DynamicCalculator(object_length=controller_object_length, object_width=controller_object_width, gravity=gravity)
        self.action_space = self.robot.action_space = gym.spaces.Discrete(6)

        self.seed(seed)             # set seed for same randomness each time
        self.reward_type, self.gravity = reward_type, gravity
        self.is_feasible_check, self.object_length, self.is_render = is_feasible_check, controller_object_length, is_render

        self.mode, self.current_step, self.flip_once, self.info_sim_time = 'f1', 0, False, 0
        self.repose_buffer, self.slide_buffer, self.flip_buffer = [], [], []
        self.level = 'easy'

        # sample data
        self.f1_file_path = os.path.join(os.path.dirname(__file__), "data", "f1")
        self.f2_file_path = os.path.join(os.path.dirname(__file__), "data", "f2")
        self.f3_file_path = os.path.join(os.path.dirname(__file__), "data", "f3")
        if not os.path.exists(self.f1_file_path): raise ValueError("{} does not exist!".format(self.f1_file_path))
        if not os.path.exists(self.f2_file_path): raise ValueError("{} does not exist!".format(self.f2_file_path))
        if not os.path.exists(self.f3_file_path): raise ValueError("{} does not exist!".format(self.f3_file_path))

    def step(self, action):
        """
        Discretize actions,
            {0,1,...,5}:  repose, +-delta_x, +-delta_y, +-delta_theta
        """
        self.current_step += 1
        assert (not self.scene.multiplayer)
        if self.is_feasible_check:
            feasibility = self.check_action_feasibility(action, self.mode)
        else:
            feasibility = True
        if feasibility:
            self.make_action(action, self.mode)
        state = self.robot.calc_state()
        reward = self.compute_reward(self.robot.object_pos, self.robot.target_pos, feasibility)
        self.done = (self.current_step >= self.max_episode_steps) or \
               self.is_success(self.robot.object_pos, self.robot.target_pos) or \
               self.is_lose_contact()
        info = {'custom':{'action': action, 'feasibility': feasibility}}
        if self.done:
            info = {'custom':{
                'is_drop': self.is_lose_contact(),
                'is_succ': self.is_success(self.robot.object_pos, self.robot.target_pos),
                'is_time_exceed': self.current_step > self.max_episode_steps,
                'reward': reward,
                'epi_len': self.current_step,
                'epi_sim_len': self.info_sim_time,       # real simulation steps
                'action': action,
                'feasibility': feasibility,
            }}
            self.current_step, self.mode, self.flip_once, self.info_sim_time = 0, 'f1', False, 0
        return state, reward, self.done, info

    def make_action(self, action, mode):
        assert action in list(range(6))
        self._repose(self.delta_repose[action], mode)
        return

    def _repose(self, delta_des, mode):
        def compute_des(x, delta_des, ds, mode):
            """For rotation, the rotation centroid is the alone finger, instead of the object origin"""
            if delta_des[2] == 0:  # x, y direction
                return delta_des + x
            d = {'f1': ds[0], 'f2': ds[1], 'f3': ds[2]}[mode]
            pos_rot_center = x + np.array([d * np.cos(x[2]), d * np.sin(x[2]), delta_des[2]])
            pos_obj_center = pos_rot_center - np.array(
                [d * np.cos(x[2] + delta_des[2]), d * np.sin(x[2] + delta_des[2]), 0])
            return pos_obj_center

        self.repose_buffer = []
        q, _, x, _ = self.robot.get_state()
        ds = self.controller.compute_contact_distance(x, q)
        x_des = compute_des(x, delta_des, ds, mode)
        self.update_indicator(x_des, ds)

        for i in range(self.repose_duration):
            self.info_sim_time += 1  # count sim time
            q, q_dot, x, x_dot = self.robot.get_state()
            ds = self.controller.compute_contact_distance(x, q)
            # Inverse Dynamic controller (operational space)
            tau_repose = self.controller.inverse_dynamic_controller_operational_space(
                mode=mode, pose_des=x_des, pose_cur=x, pose_cur_dot=x_dot,
                q=q, qdot=q_dot, d1=ds[0], d2=ds[1], d3=ds[2])
            self.robot.apply_action(tau_repose, clip=False)
            self.scene.global_step()

            if i % 2 and self.is_render:
                frame = self.render('rgb_array').astype('uint8')
                self.repose_buffer.append(frame)
        return

    def check_action_feasibility(self, action, mode):
        """check reposing action feasibility"""
        assert action in list(range(6))
        q, _, x, _ = self.robot.get_state()
        d1, d2, d3 = self.controller.compute_contact_distance(x, q)
        x_des = self.delta_repose[action] + x
        q = self.controller.compute_IK(x_des, d1, d2, d3, mode)
        return q is not None


class Gripper2DPrimitiveDiscreteReposeSlideEnv(Gripper2DPrimitiveDiscreteReposeEnv):
    """
     Three finger env -- Midium
    12 discrete actions, 18 observation dim
    """
    def __init__(self, urdf_name="three_fingers_pole_indicator.urdf", render=False, measurement_err=False,
                 hide_target=False, hide_target_local=False, init_file=None, reward_type='sparse', seed=0,
                 is_feasible_check=True, controller_object_length=0.5, controller_object_width=0.02, gravity=0.0, is_render=False,
                 cam_dist=CAM_DIST
                 ):

        super().__init__(urdf_name=urdf_name, render=render, measurement_err=measurement_err, hide_target=hide_target,
                         hide_target_local=hide_target_local, init_file=init_file, reward_type=reward_type, seed=seed,
                         is_feasible_check=is_feasible_check, controller_object_length=controller_object_length,
                         controller_object_width=controller_object_width, gravity=gravity, is_render=is_render,
                         cam_dist=cam_dist)
        self.action_space = self.robot.action_space = gym.spaces.Discrete(12)
        self.level = 'medium'

    def make_action(self, action, mode):
        assert action in list(range(12))
        if action < 6:
            self._repose(self.delta_repose[action], mode)
        else:
            self._slide(self.delta_slide[action-6], mode)

    def check_action_feasibility(self, action, mode):
        if action < 6:
            return super().check_action_feasibility(action, mode)
        else:   # slide
            q, _, x, _ = self.robot.get_state()
            ds = self.controller.compute_contact_distance(x, q)
            ds_des = self.delta_slide[action - 6] + ds
            q = self.controller.compute_IK(x, *ds_des, mode)
            return q is not None

    def _slide(self, delta_slide, mode):
        self.slide_buffer = []
        q, _, x, _ = self.robot.get_state()
        finger = delta_slide.nonzero()
        assert len(finger) == 1, 'only accepts sliding action on one finger, received action {delta_slide}'
        finger = finger[0][0]
        move_joints = [finger * 2, finger * 2 + 1]  # finger 1 need fix other two fingers
        fix_joints = [x for x in [0, 1, 2, 3, 4, 5] if x not in move_joints]
        self._p.setJointMotorControlArray(0, jointIndices=fix_joints,
                                          controlMode=p.VELOCITY_CONTROL, forces=[100] * 4)

        q_des, _, x_des, _ = self.robot.get_state()
        ds = self.controller.compute_contact_distance(x_des, q_des)
        d, d_fix = ds[finger], ds[[x for x in [0, 1, 2] if x not in [finger]]]
        d_des = delta_slide[finger] + d
        ddes = np.linspace(d, d_des, self.slide_duration)

        ds = self.controller.compute_contact_distance(x_des, q_des)
        # set indicator position
        self.update_indicator(x_des, ds)
        self.robot.target_contact_point_joints[finger].set_state(d_des, 0)

        for i in range(self.slide_duration):
            self.info_sim_time += 1
            q, q_dot, x, x_dot = self.robot.get_state()
            ds = self.controller.compute_contact_distance(x, q)
            tau_slide = self.controller.slide_torque(
                mode=mode, finger=finger, pose_des=x_des, pose_cur=x, pose_dot=x_dot,
                q_des=q_des, q=q, qdot=q_dot, d1=ds[0], d2=ds[1], d3=ds[2], d_des=ddes[i])
            if tau_slide is None: break  # target point out of finger workspace
            self.robot.apply_action(tau_slide, clip=False)
            self.scene.global_step()

            if i % 2 and self.is_render:
                frame = self.render('rgb_array').astype('uint8')
                self.slide_buffer.append(frame)

        self._p.setJointMotorControlArray(0, jointIndices=[0, 1, 2, 3, 4, 5],
                                          controlMode=p.VELOCITY_CONTROL, forces=[0] * 6)
        return


class Gripper2DPrimitiveDiscreteFullEnvV1(Gripper2DPrimitiveDiscreteReposeSlideEnv):
    """
     Three finger env -- hard
    14 discrete actions, 18 observation dim
    """
    def __init__(self, urdf_name="three_fingers_pole_indicator.urdf", render=False, measurement_err=False,
                 hide_target=False, hide_target_local=False, init_file=None, reward_type='sparse', seed=0,
                 is_feasible_check=True, controller_object_length=0.5, controller_object_width=0.02, gravity=0.0,
                 is_render=False,
                 ):
        super().__init__(urdf_name=urdf_name, render=render, measurement_err=measurement_err, hide_target=hide_target,
                         hide_target_local=hide_target_local, init_file=init_file, reward_type=reward_type, seed=seed,
                         is_feasible_check=is_feasible_check, controller_object_length=controller_object_length,
                         controller_object_width=controller_object_width, gravity=gravity, is_render=is_render)
        self.action_space = self.robot.action_space = gym.spaces.Discrete(14)
        self.level = 'hard'
        self.flip_done = False

    def make_action(self, action, mode):
        assert action in list(range(14))
        if action < 12:
            super().make_action(action, mode)
        else:
            self._flip(action)

    def check_action_feasibility(self, action, mode):
        if action < 12:
            return super().check_action_feasibility(action, mode)
        else:  # flip
            if mode != 'f1':
                return False
            q, _, x, _ = self.robot.get_state()
            d1, d2, d3 = self.controller.compute_contact_distance(x, q)
            if self.need_flip_f2 and not self.flip_done and action == 12: # and (-0.05 <= d1 - d2 <= 0.05):
                return True
            if self.need_flip_f3 and not self.flip_done and action == 13: # and (-0.05 <= d3 - d1 <= 0.05):
                return True
            return False

    def _flip(self, a):
        self.flip_buffer = []
        if a == 12:
            goal_mode, moving_finger, link_ori_flag, object_tip_relation = 'f2', 2, 1, -1
            flip_joint, post_slide_delta = [4,5], [-0.08, 0, 0]
            target_joint1, target_joint2, target_joint3 = [-np.pi / 12., np.pi * 0.9], [-np.pi/12., 1.54], [1.41, 1.56]   # hard code
            offset_x, offset_y= -self.controller.finger3_x, self.controller.arena_h/2. # used for convert world frame to local
            w, l = self.controller.w_target, 0.18
            for _ in range(5):    # pre-sliding
                q, q_dot, x, x_dot = self.robot.get_state()
                d1, d2, d3 = self.controller.compute_contact_distance(x, q)
                self._slide(np.array([0, d1 - d2 - 0.01, 0]), 'f1')
                self.flip_buffer.extend(self.slide_buffer)
                if abs(d1 - d2) < 0.03: break
        elif a == 13:
            goal_mode, moving_finger, link_ori_flag, object_tip_relation = 'f3', 1, -1, -1
            flip_joint, post_slide_delta = [2,3], [0.08, 0, 0]
            target_joint1, target_joint2, target_joint3 = [np.pi * (1 + 1./12), -np.pi * 0.9], [np.pi*(1 + 1/12.), 0], [2.0, -1]   # hard code
            offset_x, offset_y = -self.controller.finger2_x, self.controller.arena_h/2. # used for convert world frame to local
            w, l = self.controller.w_target, -0.18
            for _ in range(5):
                q, q_dot, x, x_dot = self.robot.get_state()
                d1, d2, d3 = self.controller.compute_contact_distance(x, q)
                self._slide(np.array([0, 0, d1 - d3 + 0.01]), 'f1')
                self.flip_buffer.extend(self.slide_buffer)
                if abs(d1 - d3) < 0.03: break
        else:
            raise ValueError("flip action {} is not in 12,13!".format(a))

        contact_cnt = 0
        for i in range(self.flip_duration):
            self.info_sim_time += 1
            q, q_dot, x, x_dot = self.robot.get_state()
            d1, d2, d3 = self.controller.compute_contact_distance(x, q)

            # Inverse Dynamic controller using two fingers
            tau_flip = self.controller.inverse_dynamic_controller_operational_space_2_fingers(
                pose_cur=x,
                pose_cur_dot=x_dot,
                q=q,
                qdot=q_dot,
                d1=d1, d2=d2, d3=d3, mode=self.mode, flip_joint=flip_joint
            )
            self.robot.apply_action(tau_flip, clip=False)
            if i < 100:
                self.robot._p.setJointMotorControlArray(0, jointIndices=flip_joint,
                                                        controlMode=p.POSITION_CONTROL, targetPositions=target_joint1,
                                                        forces=[1, 1])
            elif i < 200:
                self.robot._p.setJointMotorControlArray(0, jointIndices=flip_joint,
                                                        controlMode=p.POSITION_CONTROL, targetPositions=target_joint2,
                                                        forces=[1, 1])
            elif i < 300:
                self.robot._p.setJointMotorControlArray(0, jointIndices=flip_joint,
                                                        controlMode=p.POSITION_CONTROL, targetPositions=target_joint3,
                                                        forces=[0.5, 0.5])
            else:
                _,_, x, _ = self.robot.get_state()
                theta = x[2]
                target_pos = [x[0] - w / 2.0 * np.sin(theta) + l * np.cos(theta),
                              x[1] + w / 2.0 * np.cos(theta) + l * np.sin(theta)]   # world frame
                target_pos[0] += offset_x  # world to local
                target_pos[1] += offset_y
                target_joint_angle = self.controller._compute_IK_one_finger(target_pos, theta, link_ori_flag=link_ori_flag, object_tip_relation=object_tip_relation)
                if target_joint_angle is not None:
                    self.robot._p.setJointMotorControlArray(0, jointIndices=flip_joint,
                                                            controlMode=p.POSITION_CONTROL,
                                                            targetPositions=target_joint_angle,
                                                            forces=[0.1, 0.1])
                d1, d2, d3 = self.robot.get_contact_pts(self.controller.compute_forward_kinematics(self.robot.get_state()[0]))
                if None not in [d1, d2, d3]:
                    contact_cnt += 1
                    if contact_cnt > 20 and np.abs([d1, d2, d3][moving_finger]) < self.object_length/2. - 0.01:
                        if DEBUG: print("contact cnt {}, ds {},{},{}".format(contact_cnt, d1,d2,d3))
                        break
                else: contact_cnt = 0

            self.scene.global_step()
            self.robot._p.setJointMotorControlArray(0, jointIndices=[x for x in range(6)],
                                                    controlMode=p.VELOCITY_CONTROL, forces=[0]*6)
            if i % 2 and self.is_render:
                frame = self.render('rgb_array').astype('uint8')
                self.flip_buffer.append(frame)

        q, _, x, _ = self.robot.get_state()
        if a == 12:
            self.flip_done = (q[4] + q[5] >= np.pi + x[2] or x[2] < -0.3) and self.need_flip_f2
        elif a == 13:
            self.flip_done = (q[4] + q[5] <= np.pi + x[2] or x[2] > 0.3) and self.need_flip_f3
        if self.flip_done:
            self.mode = goal_mode
            self._slide(np.array(post_slide_delta), goal_mode)
            self.flip_once = True
            self.flip_buffer.extend(self.slide_buffer)
        self.robot._p.setJointMotorControlArray(0, jointIndices=flip_joint,
                                    controlMode=p.VELOCITY_CONTROL, forces=[0,0])
        return


if __name__ == '__main__':
    rng = np.random.RandomState(0)
    env = Gripper2DPrimitiveDiscreteReposeEnv(render=True, gravity=-9.8)
    s = env.reset()
    for i in range(8000):
        act = rng.randint(6)
        s,rew,done,info = env.step(act)

        if done:
            print(info)
            if info['custom']['epi_len'] == 200:
                print("time exceeds")
            elif info['custom']['is_succ']:
                print("success")
            env.reset()
