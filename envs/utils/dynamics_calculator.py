"""
Dynamic Calulator class for three fingers manipulator, including dynamic equation and
multiple controller.
"""
import numpy as np
np.warnings.filterwarnings('ignore')

class DynamicCalculator(object):
    def __init__(self, object_length=0.5, object_width=0.02, object_mass=0.01,
                 link_length_1=0.15, link_mass_1=0.1, link_length_2=0.1, link_mass_2=0.1, link_width=0.02, link_thickness=0.005,
                 x_1=0, x_2=-0.1, x_3=0.1, arena_w=0.6, arena_h=0.4, gravity=0
                 ):
        """
        Init parameters including link mass, link length, link width ...
        This part should be consistent with urdf file.
        """
        self.l1, self.l2, self.l = link_length_1, link_length_2, object_length      # link length and target length
        self.m1, self.m2, self.mass = link_mass_1, link_mass_2, object_mass         # link mass and target mass
        self.w_link, self.t_link = link_width, link_thickness                       # link width, link thickness
        self.w_target = object_width                                                # target width
        self.arena_w, self.arena_h = arena_w, arena_h                               # width and height of arena
        self.finger1_x, self.finger2_x, self.finger3_x = x_1, x_2, x_3              # x coordinates of three fingers
        self.gravity = gravity

        # PD controller
        kp_pd, kv_pd = 1,0.5
        self.Kp_PD = np.diag([kp_pd]*6)
        self.Kv_PD = np.diag([kv_pd]*6)

        # Inverse Dynamic Controller (joint space)
        kp_js, kv_js = 200, 50
        self.Kp_ID_joint_space = np.diag([kp_js]*6)
        self.Kv_ID_joint_space = np.diag([kv_js]*6)

        # Operational Space
        kp_os, kv_os = 100, 30
        self.Kp_ID_operation_space = np.diag([kp_os]*9)
        self.Kv_ID_operation_space = np.diag([kv_os]*9)

        # sliding
        kp_sliding, kv_sliding = 200, 50
        self.Kp_sliding = np.diag([kp_sliding]*9)
        self.Kv_sliding = np.diag([kv_sliding]*9)

    def compute_contact_distance(self, pose, q):
        """Given object pose and joint angles, return contact distance d1, d2, d3"""
        # compute end (not tip) position
        end1_local = self._compute_FK(q[0], q[1], l2=self.l2-self.w_link/2.)
        end2_local = self._compute_FK(q[2], q[3], l2=self.l2-self.w_link/2.)
        end3_local = self._compute_FK(q[4], q[5], l2=self.l2-self.w_link/2.)
        end1_world, end2_world, end3_world = self._local_frame_to_world_frame(end1_local, end2_local, end3_local)
        ds = [None, None, None]
        for i, end in enumerate([end1_world, end2_world, end3_world]):
            dist = np.sqrt(max((end[0] - pose[0]) ** 2 + \
                                (end[1] - pose[1]) ** 2 - (self.w_link/2. + self.w_target/2.)**2, 0))
            if dist > self.l/2.: dist = self.l/2.
            # determine the +/-, compute the two possible ct locations in the pole
            pt_pos = np.array([pose[0] + dist * np.cos(pose[2]),
                               pose[1] + dist * np.sin(pose[2])])
            pt_neg = np.array([pose[0] - dist * np.cos(pose[2]),
                               pose[1] - dist * np.sin(pose[2])])
            dist_list = [np.linalg.norm(pt_pos - end), np.linalg.norm(pt_neg - end)]
            ds[i] = dist if dist_list[0] < dist_list[1] else -dist
        return np.array(ds)

    def compute_forward_kinematics(self, q):
        """Given all joint angles, return the tip position in world frame"""
        assert len(q) == 6
        p1_local = self._compute_FK(q[0], q[1])
        p2_local = self._compute_FK(q[2], q[3])
        p3_local = self._compute_FK(q[4], q[5])
        # local frame to world frame
        p1_world, p2_world, p3_world = self._local_frame_to_world_frame(p1_local, p2_local, p3_local)
        return p1_world, p2_world, p3_world

    def compute_IK(self, pose, d1, d2, d3, mode='f1', search=False):
        """Given pole pose, compute joint configurations and mode, compute joint angles"""
        assert len(pose) == 3
        c1_world, c2_world, c3_world = self._get_contact_points(pose, d1, d2, d3, mode)
        c1_local, c2_local, c3_local = self._world_frame_to_local_frame(c1_world, c2_world, c3_world)

        # Compute joint angles for each finger
        object_tip_relation_f2 = -1 if mode == 'f3' else 1
        object_tip_relation_f3 = -1 if mode == 'f2' else 1
        q1 = self._compute_IK_one_finger(c1_local, pose[2], link_ori_flag=1, object_tip_relation=1)
        q2 = self._compute_IK_one_finger(c2_local, pose[2], link_ori_flag=-1, object_tip_relation=object_tip_relation_f2)
        q3 = self._compute_IK_one_finger(c3_local, pose[2], link_ori_flag=1, object_tip_relation=object_tip_relation_f3)
        if search and None in [q1, q2, q3]:
            return [[q1, q2, q3].index(None)]
        # if out of workspace, return none
        if None in [q1, q2, q3]:
            return None
        return np.array([q1[0], q1[1], q2[0], q2[1], q3[0], q3[1]])

    def PD_controllor(self, pose_des, pose_cur, q, qdot, d1, d2, d3):
        "UNDER FURTHER TEST"
        "Given a desired pole pose (x,y,theta), calculate applied torque"
        # calculate target joint angles given pole pose
        q_des = self.compute_IK(pose_des, d1, d2, d3)
        # contact force to balance the target
        balance_force = 5 * self._get_grasp_matrix_null_space_forces(d1, d2, d3)
        # compute torques to generate balance force
        J = self._get_jacobian(q, pose_cur[2])
        balance_torque = np.dot(J.T, balance_force)

        v = np.dot(self.Kp_PD, (np.array(q_des) - np.array(q))) - np.dot(self.Kv_PD, qdot)
        safe_value = 0.03
        for i in range(len(v)):
            if np.abs(v[i]) > safe_value:
                v[i] = np.sign(v[i]) * safe_value

        tau = v + balance_torque
        return tau

    def inverse_dynamic_controller_operational_space(self, pose_des, pose_cur, pose_cur_dot, q, qdot, d1, d2, d3, mode='f1'):
        """Operational space control"""
        Mhnd, V = self._get_mass_matrix(q, qdot)
        Mobj = self._get_object_mass_matrix()
        delta_x = self._delta_from_world_to_contact_frame(pose_cur, pose_des, d1, d2, d3, mode)
        J = self._get_jacobian(q, pose_cur[2], mode)
        x_vel_cur = np.dot(J, qdot)
        G = self._get_hand_matrix(pose_cur[2], d1, d2, d3, mode)
        v = np.dot(self.Kp_ID_operation_space, delta_x) - \
            np.dot(self.Kv_ID_operation_space, x_vel_cur)
        Jdot = self._get_jacobian_derivative(q, qdot, pose_cur[2], pose_cur_dot[2], mode)

        dynamic_tau = (Mhnd.dot(np.linalg.pinv(J)) + (J.T).dot(np.linalg.pinv(G)).dot(Mobj).dot(np.linalg.pinv(G.T))).dot(v) + \
                      V - Mhnd.dot(np.linalg.pinv(J)).dot(Jdot).dot(qdot)
        balance_torque = self._get_balance_torque(pose_cur, q, d1, d2, d3, mode)
        g_term = self._get_gravity_term(q)
        return dynamic_tau + balance_torque + g_term

    def inverse_dynamic_controller_operational_space_2_fingers(self, pose_cur, pose_cur_dot, q, qdot, d1, d2, d3, mode,
                                                               flip_joint):
        """Control the pole using two fingers"""
        J = self._get_jacobian(q, pose_cur[2])
        balance_force = self._get_balance_force_two_fingers(pose_cur, d1, d2, d3, flip_joint, mode)
        balance_torque = np.dot(J.T,
                                balance_force)  # a very small force apply on 2 contacting fingers, no force on flipping finger

        g_term = self._get_gravity_term(q)
        # Operation Space
        M, V = self._get_mass_matrix(q, qdot)
        # contract frames
        x_vel_cur = np.dot(J, qdot)

        kp = 100
        KP = np.diag([kp] * 9)

        v = - np.dot(KP, x_vel_cur)
        Jdot = self._get_jacobian_derivative(q, qdot, pose_cur[2], pose_cur_dot[2])
        tau = np.dot(M, np.dot(np.linalg.pinv(J), v)) + \
              V - np.dot(M, np.dot(np.linalg.pinv(J), np.dot(Jdot, qdot))) + \
              balance_torque + \
              g_term
        return tau

    def slide_torque(self, finger, pose_des, pose_cur, pose_dot, q_des, q, qdot, d1, d2, d3, d_des, mode='f1'):
        """Joint space control sliding"""
        if finger == 0:
            q_des_slide = self.compute_IK(pose_des, d_des, d2, d3, mode)
        elif finger == 1:
            q_des_slide = self.compute_IK(pose_des, d1, d_des, d3, mode)
        elif finger == 2:
            q_des_slide = self.compute_IK(pose_des, d1, d2, d_des, mode)
        else:
            raise ValueError
        if q_des_slide is None: return None
        q_des[finger*2:finger*2 + 2] = q_des_slide[finger*2:finger*2 + 2]
        Mhnd, V = self._get_mass_matrix(q, qdot)
        v = np.dot(self.Kp_ID_joint_space, (np.array(q_des) - np.array(q))) - np.dot(self.Kv_ID_joint_space, qdot)
        g_term = self._get_gravity_term(q)
        balance_torque = self._get_balance_torque(pose_cur, q, d1, d2, d3, mode)
        return Mhnd.dot(v) + V + balance_torque + g_term

    def regrasp_torque(self, pose, q, d1, d2, d3, mode):
        """Regrasp the pole if accidently lose contact"""
        balance_force = self._get_balance_force(pose, d1, d2, d3, mode)
        J = self._get_jacobian(q, pose[2], mode)
        balance_torque = J.T.dot(balance_force)
        g_term = self._get_gravity_term(q)
        return balance_torque + g_term

    def _compute_FK(self, q1, q2, l2=None):
        """Given q1, q2, return the tip position in local frame"""
        # for compute the end position instead of the tip if not None
        if l2 is None: l2 = self.l2
        x = self.l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        return [x, y]

    def _local_frame_to_world_frame(self, p1, p2, p3):
        """Given tip positions in local frames, transform into world frame"""
        assert len(p1) == 2 and len(p2) == 2 and len(p3) == 2, \
            'The length of tip points shoule be 2, but {},{},{}'.format(len(p1), len(p2), len(p3))
        x1, y1 = -p1[0], self.arena_h / 2. - p1[1]
        x2, y2 = p2[0] + self.finger2_x, p2[1] - self.arena_h / 2.
        x3, y3 = p3[0] + self.finger3_x, p3[1] - self.arena_h / 2.
        return [x1, y1], [x2, y2], [x3, y3]

    def _world_frame_to_local_frame(self, p1, p2, p3):
        """Given tip positions in world frames, transform into local frame"""
        assert len(p1) == 2 and len(p2) == 2 and len(p3) == 2, \
            'The length of tip points shoule be 2, but {},{},{}'.format(len(p1), len(p2), len(p3))
        c1 = [-p1[0] + self.finger1_x, self.arena_h / 2.0 - p1[1]]
        c2 = [p2[0] - self.finger2_x, self.arena_h / 2.0 + p2[1]]
        c3 = [p3[0] - self.finger3_x, self.arena_h / 2.0 + p3[1]]
        return c1, c2, c3

    def _get_balance_torque(self, pose, q, d1, d2, d3, mode):
        """Return the required torque to balance the pole"""
        balance_force = self._get_balance_force(pose, d1, d2, d3, mode)
        J = self._get_jacobian(q, pose[2], mode)
        balance_torque = J.T.dot(balance_force)
        return balance_torque

    def _get_grasp_matrix_null_space_forces(self, d1, d2, d3, mode='f1'):
        # return forces in grasp matrix null space
        if mode == 'f1':
            null_space = np.array([d3-d2, 0, 0, d3-d1, 0, 0, d1-d2, 0, 0])
        elif mode == 'f2':
            null_space = np.array([d3-d2, 0, 0, d3-d1, 0, 0, d2-d1, 0, 0])
        elif mode == 'f3':
            null_space = np.array([d3-d2, 0, 0, d1-d3, 0, 0, d1-d2, 0, 0])
        else:
            raise ValueError
        return null_space

    def _get_mass_matrix(self, q, qdot):
        """
        Return the mass matrix and Coriolis term
        """
        assert len(q) == 6 and len(qdot) == 6

        Iz1 = 1 / 12.0 * self.m1 * (self.l1 ** 2 + self.w_link ** 2)
        Iz2 = 1 / 12.0 * self.m2 * (self.l2 ** 2 + self.w_link ** 2)
        alpha = Iz1 + Iz2 + self.m1 * self.l1 ** 2 / 4. + self.m2 * (self.l1 ** 2 + self.l2 ** 2 / 4.)
        beta = self.m2 * self.l1 * self.l2 / 2.
        delta = Iz2 + self.m2 * self.l2 ** 2 / 4.

        M = np.array([
            [alpha + 2 * beta * np.cos(q[1]), delta + beta * np.cos(q[1]), 0, 0, 0, 0],
            [delta + beta * np.cos(q[1]), delta, 0, 0, 0, 0],
            [0, 0, alpha + 2 * beta * np.cos(q[3]), delta + beta * np.cos(q[3]), 0, 0],
            [0, 0, delta + beta * np.cos(q[3]), delta, 0, 0],
            [0, 0, 0, 0, alpha + 2 * beta * np.cos(q[5]), delta + beta * np.cos(q[5])],
            [0, 0, 0, 0, delta + beta * np.cos(q[5]), delta],
        ])

        V = np.dot(np.array([
            [-beta * np.sin(q[1]) * qdot[1], -beta * np.sin(q[1]) * (qdot[0] + qdot[1]), 0, 0, 0, 0],
            [beta * np.sin(q[1]) * qdot[0], 0, 0, 0, 0, 0],
            [0, 0, -beta * np.sin(q[3]) * qdot[3], -beta * np.sin(q[3]) * (qdot[2] + qdot[3]), 0, 0],
            [0, 0, beta * np.sin(q[3]) * qdot[2], 0, 0, 0],
            [0, 0, 0, 0, -beta * np.sin(q[5]) * qdot[5], -beta * np.sin(q[5]) * (qdot[4] + qdot[5])],
            [0, 0, 0, 0, beta * np.sin(q[5]) * qdot[4], 0],
        ]), qdot)

        return M, V

    def _get_hand_matrix(self, theta, d1, d2, d3, mode='f1'):
        """
        Given orientation of pole, return hand matrix.
        """
        if mode == 'f1':
            G = np.array([
                [np.sin(theta), -np.cos(theta), - d1],
                [np.cos(theta), np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [-np.sin(theta), np.cos(theta), d2],
                [-np.cos(theta), -np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [-np.sin(theta), np.cos(theta), d3],
                [-np.cos(theta), -np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
            ]).T
        elif mode == 'f2':
            G = np.array([
                [np.sin(theta), -np.cos(theta), -d1],
                [np.cos(theta), np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [-np.sin(theta), np.cos(theta), d2],
                [-np.cos(theta), -np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [np.sin(theta), -np.cos(theta), -d3],
                [np.cos(theta), np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
            ]).T
        elif mode == 'f3':
            G = np.array([
                [np.sin(theta), -np.cos(theta), -d1],
                [np.cos(theta), np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [np.sin(theta), -np.cos(theta), -d2],
                [np.cos(theta), np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [-np.sin(theta), np.cos(theta), d3],
                [-np.cos(theta), -np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
            ]).T
        return G.copy()

    def _get_jacobian(self, q, theta, mode="f1"):
        assert len(q) == 6

        q11, q12, q21, q22, q31, q32 = q[0], q[1], q[2], q[3], q[4], q[5]
        J = np.array([
            [self.l1 * np.cos(q11 - theta) + (self.l2 - self.w_link / 2.0) * np.cos(q11 + q12 - theta),
             (self.l2 - self.w_link / 2.0) * np.cos(q11 + q12 - theta), 0,
             0, 0, 0],
            [self.l1 * np.sin(q11 - theta) + (self.l2 - self.w_link / 2.0) * np.sin(q11 + q12 - theta),
             (self.l2 - self.w_link / 2.0) * np.sin(q11 + q12 - theta), 0,
             0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, self.l1 * np.cos(q21 - theta) + (self.l2 - self.w_link / 2.0) * np.cos(q21 + q22 - theta),
             (self.l2 - self.w_link / 2.0) * np.cos(q21 + q22 - theta), 0, 0],
            [0, 0, self.l1 * np.sin(q21 - theta) + (self.l2 - self.w_link / 2.0) * np.sin(q21 + q22 - theta),
             (self.l2 - self.w_link / 2.0) * np.sin(q21 + q22 - theta), 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, self.l1 * np.cos(q31 - theta) + (self.l2 - self.w_link / 2.0) * np.cos(q31 + q32 - theta),
             (self.l2 - self.w_link / 2.0) * np.cos(q31 + q32 - theta)],
            [0, 0, 0, 0, self.l1 * np.sin(q31 - theta) + (self.l2 - self.w_link / 2.0) * np.sin(q31 + q32 - theta),
             (self.l2 - self.w_link / 2.0) * np.sin(q31 + q32 - theta)],
            [0, 0, 0, 0, 0, 0]
        ])

        if mode == "f2":
            J[-3] *= -1
            J[-2] *= -1
        if mode == "f3":
            J[3] *= -1
            J[4] *= -1
        return J

    def _get_jacobian_derivative(self, q, qdot, theta, theta_dot, mode="f1"):
        assert len(q) == 6 and len(qdot) == 6

        q11, q12, q21, q22, q31, q32 = q[0], q[1], q[2], q[3], q[4], q[5]
        qdot11, qdot12, qdot21, qdot22, qdot31, qdot32 = qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5]
        Jdot = np.array([
            [-self.l1 * np.sin(q11 - theta) * (qdot11 - theta_dot) - (self.l2 - self.w_link / 2.0) * np.sin(q11 + q12 - theta) * (
                    qdot11 + qdot12 - theta_dot),
             -(self.l2 - self.w_link / 2.0) * np.sin(q11 + q12 - theta) * (qdot11 + qdot12 - theta_dot), 0, 0, 0, 0],
            [self.l1 * np.cos(q11 - theta) * (qdot11 - theta_dot) + (self.l2 - self.w_link / 2.0) * np.cos(q11 + q12 - theta) * (
                    qdot11 + qdot12 - theta_dot),
             (self.l2 - self.w_link / 2.0) * np.cos(q11 + q12 - theta) * (qdot11 + qdot12 - theta_dot), 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, -self.l1 * np.sin(q21 - theta) * (qdot21 - theta_dot) - (self.l2 - self.w_link / 2.0) * np.sin(q21 + q22 - theta) * (
                    qdot21 + qdot22 - theta_dot),
             -(self.l2 - self.w_link / 2.0) * np.sin(q21 + q22 - theta) * (qdot21 + qdot22 - theta_dot), 0, 0],
            [0, 0, self.l1 * np.cos(q21 - theta) * (qdot21 - theta_dot) + (self.l2 - self.w_link / 2.0) * np.cos(q21 + q22 - theta) * (
                    qdot21 + qdot22 - theta_dot),
             (self.l2 - self.w_link / 2.0) * np.cos(q21 + q22 - theta) * (qdot21 + qdot22 - theta_dot), 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0,
             -self.l1 * np.sin(q31 - theta) * (qdot31 - theta_dot) - (self.l2 - self.w_link / 2.0) * np.sin(q31 + q32 - theta) * (
                     qdot31 + qdot32 - theta_dot),
             -(self.l2 - self.w_link / 2.0) * np.sin(q31 + q32 - theta) * (qdot31 + qdot32 - theta_dot)],
            [0, 0, 0, 0,
             self.l1 * np.cos(q31 - theta) * (qdot31 - theta_dot) + (self.l2 - self.w_link / 2.0) * np.cos(q31 + q32 - theta) * (
                     qdot31 + qdot32 - theta_dot),
             (self.l2 - self.w_link / 2.0) * np.cos(q31 + q32 - theta) * (qdot31 + qdot32 - theta_dot)],
            [0, 0, 0, 0, 0, 0]
        ])

        if mode == "f2":
            Jdot[-3] *= -1
            Jdot[-2] *= -1
        if mode == "f3":
            Jdot[3] *= -1
            Jdot[4] *= -1
        return Jdot

    def _compute_IK_one_finger(self, p_contact, theta, link_ori_flag=1, object_tip_relation=1):
        """
        p: the actual contact point, in local frame
        theta: orientation of object
        link_ori_flag: for finger 2, the orientation of distal finger is opposite to other two fingers, 1 for f1,f3, -1 for f2
        object_tip_relation: 1 for object is above the finger tip, -1 for object is under the finger tip
        """
        p_center = [p_contact[0] + object_tip_relation * self.w_link / 2.0 * np.sin(theta),
                    p_contact[1] - object_tip_relation * self.w_link / 2.0 * np.cos(theta)]
        l2 = self.l2 - self.w_link/2.0

        q2 = link_ori_flag * np.arccos((p_center[0] ** 2 + p_center[1] ** 2 - self.l1 ** 2 - l2 ** 2) / (2 * self.l1 * l2))
        q1 = np.arctan(p_center[1] / (p_center[0] + 0.0001)) - np.arctan(
            l2 * np.sin(q2) / (self.l1 + l2 * np.cos(q2)))
        q1 = q1 + np.pi if q1 < 0 else q1
        if np.isnan(q1) or np.isnan(q2):
            return None
        return [q1, q2]

    def _get_contact_points(self, pose, d1, d2, d3, mode='f1'):
        """Given pole pose, compute contact point locations in world frame"""
        assert len(pose) == 3
        x, y, theta = pose
        if mode == 'f1':    # local frame pts
            c1_local, c2_local, c3_local = [d1, self.w_target / 2.], [d2, -self.w_target / 2.], [d3, -self.w_target / 2.]
        elif mode == 'f2':
            c1_local, c2_local, c3_local = [d1, self.w_target / 2.], [d2, -self.w_target / 2.], [d3, self.w_target / 2.]
        elif mode == 'f3':
            c1_local, c2_local, c3_local = [d1, self.w_target / 2.], [d2, self.w_target / 2.], [d3, -self.w_target / 2.]
        else:
            raise ValueError
        R_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        center = np.array([x, y])
        mul_add = lambda c_local: R_matrix.dot(c_local) + center
        c1_world, c2_world, c3_world = mul_add(c1_local), mul_add(c2_local), mul_add(c3_local)
        return [c1_world, c2_world, c3_world]

    def _get_object_mass_matrix(self):
        M = np.diag([self.mass, self.mass, 1 / 12. * self.mass * (self.w_target ** 2 + self.l ** 2)])
        return M

    def _get_gravity_term(self, q):
        """The gravity term of manipulations (not the object)"""
        g = self.gravity
        g_term = -np.array([
            - self.m1 * self.l1 / 2. * g * np.cos(q[0]) - self.m2 * g * (
                self.l2 / 2. * np.cos(q[0] + q[1]) + self.l1 * np.cos(q[0])),
            - self.m2 * self.l2 / 2. * g * np.cos(q[0] + q[1]),
            self.m1 * self.l1 / 2. * g * np.cos(q[2]) + self.m2 * g * (
                self.l2 / 2. * np.cos(q[2] + q[3]) + self.l1 * np.cos(q[2])),
            self.m2 * self.l2 / 2. * g * np.cos(q[2] + q[3]),
            self.m1 * self.l1 / 2. * g * np.cos(q[4]) + self.m2 * g * (
                self.l2 / 2. * np.cos(q[4] + q[5]) + self.l1 * np.cos(q[4])),
            self.m2 * self.l2 / 2. * g * np.cos(q[4] + q[5]),
        ])

        return g_term

    def _delta_from_world_to_contact_frame(self, pose_cur, pose_des, d1, d2, d3, mode='f1'):
        """Transform delta x from world frame to contact frames"""
        c1, c2, c3 = self._get_contact_points(pose_cur, d1, d2, d3, mode)
        c1_des, c2_des, c3_des = self._get_contact_points(pose_des, d1, d2, d3, mode)
        if mode == 'f1':
            x_cur = [-c1[1], c1[0], pose_cur[2], c2[1], -c2[0], pose_cur[2], c3[1], -c3[0], pose_cur[2]]
            x_des = [-c1_des[1], c1_des[0], pose_des[2], c2_des[1], -c2_des[0], pose_des[2], c3_des[1], -c3_des[0],
                     pose_des[2]]
        elif mode == 'f2':
            x_cur = [-c1[1], c1[0], pose_cur[2], c2[1], -c2[0], pose_cur[2], -c3[1], c3[0], pose_cur[2]]
            x_des = [-c1_des[1], c1_des[0], pose_des[2], c2_des[1], -c2_des[0], pose_des[2], -c3_des[1], c3_des[0],
                     pose_des[2]]
        elif mode == 'f3':
            x_cur = [-c1[1], c1[0], pose_cur[2], -c2[1], c2[0], pose_cur[2], c3[1], -c3[0], pose_cur[2]]
            x_des = [-c1_des[1], c1_des[0], pose_des[2], -c2_des[1], c2_des[0], pose_des[2], c3_des[1], -c3_des[0],
                     pose_des[2]]
        else:
            raise ValueError
        return np.array(x_des) - np.array(x_cur)

    def _get_balance_force(self, pose_cur, d1, d2, d3, mode):
        # contact force to balance the target
        G = self._get_hand_matrix(pose_cur[2], d1, d2, d3, mode)
        balance_force = -np.dot(np.linalg.pinv(G), np.array([0, self.gravity * self.mass, 0]))
        null_force = self._get_grasp_matrix_null_space_forces(d1, d2, d3, mode)
        i = 0
        while balance_force[0] <= 0.01 or balance_force[3] <= 0.01 or balance_force[6] <= 0.01 or \
            np.abs(balance_force[1]) > 0.25 * balance_force[0] or \
            np.abs(balance_force[4]) > 0.25 * balance_force[3] or \
            np.abs(balance_force[7]) > 0.25 * balance_force[6]:
            balance_force += null_force
            i += 1
            if i > 20: break
        return balance_force

    def _get_balance_force_two_fingers(self, pose_cur, d1, d2, d3, flip_joint, mode):
        """in local frame"""
        return [0.5, 0, 0, 0.5, 0, 0, 0, 0, 0] if 5 in flip_joint else [0.5, 0, 0, 0, 0, 0, 0.5, 0, 0]

    def compute_pole_tau(self, pddot):
        "TEST PURPOSE"
        "Given pole pose (x,y,z), compute the force needed to follow its trajectory"
        assert len(pddot) == 3
        g = self.gravity
        M = self._get_object_mass_matrix()
        force = np.dot(M, pddot) + np.array([0, -g*self.mass, 0])
        return force

    def compute_dynamic_tau_joint_space(self, q, qdot, qddot):
        "TEST PURPOSE"
        "Joint space control, compute the torque for manipulator without contacting objects"
        assert len(q) == 6 and len(qdot) == 6 and len(qddot) == 6
        g_term = self._get_gravity_term(q)
        M, V= self._get_mass_matrix(q, qdot)
        tau = np.dot(M, qddot) + V + g_term
        return tau

class DynamicCalculatorCube(DynamicCalculator):
    def __init__(self, object_length=0.1, object_width=0.1, object_mass=0.01,
                 link_length_1=0.125, link_mass_1=0.1, link_length_2=0.1, link_mass_2=0.1, link_width=0.02, link_thickness=0.005,
                 x_1=0, x_2=-0.05, x_3=0.05, arena_w=0.6, arena_h=0.4, gravity=0):
        super().__init__(object_length, object_width, object_mass, link_length_1, link_mass_1, link_length_2, link_mass_2,
                         link_width, link_thickness, x_1, x_2, x_3, arena_w, arena_h, gravity)

    def respose_control(self, pose_des, pose_cur, pose_cur_dot, q, qdot, d1, d2, d3, mode='f1'):
        """f1, f2, f3, f4, f5 modes"""
        if mode == "f1":
            return super().inverse_dynamic_controller_operational_space(pose_des, pose_cur, pose_cur_dot, q, qdot, d1, d2, d3, "f1")
        else:
            return self.idc_two_fingers(pose_des, pose_cur, pose_cur_dot, q, qdot, d1, d2, d3, mode)

    def idc_two_fingers(self, pose_des, pose_cur, pose_cur_dot, q, qdot, d1, d2, d3, mode):
        mode_old = {"f2": "f1", "f3": "f2", "f4":"f1", "f5":"f3"}[mode]
        Mhnd, V = self._get_mass_matrix_2(q, qdot, mode)
        Mobj = self._get_object_mass_matrix()
        J, Jdot = self._get_jacobian_jderivative_2(q, qdot, pose_cur[2], pose_cur_dot[2], mode, mode_old)
        v = self._get_op_v_2(q, qdot, pose_cur, pose_des, d1, d2, d3, mode, mode_old)
        G = self._get_hand_matrix_2(pose_cur[2], d1, d2, d3, mode, mode_old)
        balance_force = self._get_balance_force_2(mode)
        balance_torque = np.dot(J.T, balance_force)
        g_term = self._get_gravity_term_2(q, mode)
        qdot = self._get_qdot_2(qdot, mode)

        dynamic_tau = (Mhnd.dot(np.linalg.pinv(J)) + (J.T).dot(np.linalg.pinv(G)).dot(Mobj).dot(
            np.linalg.pinv(G.T))).dot(v) + \
                      V - Mhnd.dot(np.linalg.pinv(J)).dot(Jdot).dot(qdot)
        tau = dynamic_tau + balance_torque + g_term

        tau_idx = {"f2":[0,1,2,3], "f3":[2,3,4,5], "f4":[0,1,4,5], "f5":[2,3,4,5]}[mode]
        tau_list = np.zeros(6)
        tau_list[tau_idx] = tau
        return tau_list

    def _get_mass_matrix_2(self, q, qdot, mode):
        Mhnd, V = super()._get_mass_matrix(q, qdot)
        idx = {"f2": [0,1,2,3], "f3": [2,3,4,5], "f4": [0,1,4,5], "f5": [2,3,4,5]}[mode]
        Mhnd, V = Mhnd[idx][:, idx], V[idx]
        return Mhnd, V
    def _get_jacobian_jderivative_2(self, q, qdot, theta, theta_dot, mode, mode_old):
        J = super()._get_jacobian(q, theta, mode_old)
        Jdot = super()._get_jacobian_derivative(q, qdot, theta, theta_dot, mode_old)
        idx1 = {"f2": list(range(6)), "f3": list(range(3,9)), "f4": [0,1,2,6,7,8], "f5": list(range(3,9))}[mode]
        idx2 = {"f2": [0, 1, 2, 3], "f3": [2, 3, 4, 5], "f4": [0, 1, 4, 5], "f5": [2, 3, 4, 5]}[mode]
        J, Jdot = J[idx1][:, idx2], Jdot[idx1][:, idx2]
        return J, Jdot
    def _get_op_v_2(self, q, qdot, pose_cur, pose_des, d1, d2, d3, mode, mode_old):
        J = super()._get_jacobian(q, pose_cur[2], mode_old)
        delta_x = super()._delta_from_world_to_contact_frame(pose_cur, pose_des, d1, d2, d3, mode_old)
        x_vel_cur = np.dot(J, qdot)
        v = np.dot(self.Kp_ID_operation_space, delta_x) - \
            np.dot(self.Kv_ID_operation_space, x_vel_cur)
        idx = {"f2": list(range(6)), "f3": list(range(3,9)), "f4": [0,1,2,6,7,8], "f5": list(range(3,9))}[mode]
        v = v[idx]
        return v
    def _get_hand_matrix_2(self, theta, d1, d2, d3, mode, mode_old):
        G = super()._get_hand_matrix(theta, d1, d2, d3, mode_old)
        idx = {"f2": list(range(6)), "f3": list(range(3,9)), "f4": [0,1,2,6,7,8], "f5": list(range(3,9))}[mode]
        G = G[:, idx]
        return G
    def _get_balance_force_2(self, mode):
        if mode in ["f2", "f4"]:
            return [0.1, 0, 0, 0.2, 0, 0]
        elif mode in ["f3", "f5"]:
            return [0.3, 0.07, 0, 0.3, -0.07, 0]
    def _get_gravity_term_2(self, q, mode):
        g_term = super()._get_gravity_term(q)
        idx = {"f2": [0, 1, 2, 3], "f3": [2, 3, 4, 5], "f4": [0, 1, 4, 5], "f5": [2, 3, 4, 5]}[mode]
        return g_term[idx]
    def _get_qdot_2(self, qdot, mode):
        idx = {"f2": [0, 1, 2, 3], "f3": [2, 3, 4, 5], "f4": [0, 1, 4, 5], "f5": [2, 3, 4, 5]}[mode]
        return qdot[idx]
def main():
    controller = DynamicCalculator()
    object_pos = [0.028, -0.016, 0.194]
    d1, d2, d3 = -0.1, -0.15, 0.15
    q = controller.compute_IK(object_pos, d1, d2, d3)
    print(q)

if __name__ == '__main__':
    main()