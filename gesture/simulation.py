"""Rigid-body Panda simulation. Grasping uses contact/friction, never object attachment."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import DOWN_QUAT, array

HOME = np.array([.5, 0., .4])
REST = [0., -.4, 0., -2.2, 0., 1.8, .785398, .04, .04]


@dataclass
class RobotState:
    position: np.ndarray
    quaternion: np.ndarray
    joints: np.ndarray
    joint_velocity: np.ndarray
    grip: float
    object_position: np.ndarray
    object_quaternion: np.ndarray


class Simulation:
    def __init__(self, cfg):
        import pybullet as p
        import pybullet_data
        from pybullet_utils.bullet_client import BulletClient
        self.cfg, self.p = cfg, p
        self.client = BulletClient(connection_mode=p.DIRECT)
        self.kin = BulletClient(connection_mode=p.DIRECT)
        self.closed = False
        self.time = self.active_time = 0.
        self.ticks = self.rejections = self.active_ticks = 0
        self.reason, self.holding = "ok", False
        try:
            for client in (self.client, self.kin):
                client.setAdditionalSearchPath(pybullet_data.getDataPath())
                client.setTimeStep(1. / 240.)
                client.setPhysicsEngineParameter(numSolverIterations=120, deterministicOverlappingPairs=1)
                client.setGravity(0., 0., -9.81)
            self.floor = self.client.loadURDF("plane.urdf")
            self.kin_floor = self.kin.loadURDF("plane.urdf")
            flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            self.robot = self.client.loadURDF("franka_panda/panda.urdf", useFixedBase=True, flags=flags)
            self.kin_robot = self.kin.loadURDF("franka_panda/panda.urdf", useFixedBase=True, flags=flags)
            info = [self.client.getJointInfo(self.robot, j) for j in range(self.client.getNumJoints(self.robot))]
            self.joints = [j[0] for j in sorted(info, key=lambda j: j[3]) if j[3] >= 0]
            self.ee = next(j[0] for j in info if j[12].decode() == "panda_grasptarget")
            self.parents = {j[0]: j[16] for j in info}
            self.lower = np.array([info[j][8] for j in self.joints])
            self.upper = np.array([info[j][9] for j in self.joints])
            if len(self.joints) != 9:
                raise RuntimeError("Unexpected Panda URDF DoF layout")
            for client, body in ((self.client, self.robot), (self.kin, self.kin_robot)):
                for j, q in zip(self.joints, REST):
                    client.resetJointState(body, j, q)
            solution = self.kin.calculateInverseKinematics(
                self.kin_robot, self.ee, HOME, DOWN_QUAT, maxNumIterations=300, residualThreshold=1e-7)
            self.motor_targets = np.r_[np.clip(solution[:7], self.lower[:7], self.upper[:7]), .04, .04]
            for client, body in ((self.client, self.robot), (self.kin, self.kin_robot)):
                for j, q in zip(self.joints, self.motor_targets):
                    client.resetJointState(body, j, float(q))
            for j in self.joints[-2:]:
                self.client.changeDynamics(self.robot, j, lateralFriction=1.5,
                                           spinningFriction=.005, frictionAnchor=1)
            self.start_object = np.array([.5, -.13, .021])
            self.place_goal = np.array([.5, .18, .021])
            rng = np.random.default_rng(cfg.seed)
            self.reach_goal = np.array([.5, .12, .3]) + rng.uniform(-.015, .015, 3)
            collision = self.client.createCollisionShape(p.GEOM_BOX, halfExtents=[.02] * 3)
            visual = self.client.createVisualShape(p.GEOM_BOX, halfExtents=[.02] * 3,
                                                  rgbaColor=[.93, .38, .12, 1.])
            self.cube = self.client.createMultiBody(.06, collision, visual, self.start_object)
            self.client.changeDynamics(self.cube, -1, lateralFriction=1.2, rollingFriction=.001,
                                       spinningFriction=.002, restitution=0., linearDamping=.04,
                                       angularDamping=.04)
            self.goal_body = self._marker(self.place_goal if cfg.task == "pick-place" else self.reach_goal)
            self._motors()
            for _ in range(120):
                self.client.stepSimulation()
            self.lifted = self.success = False
            self.success_time = None
            self.dwell = 0.
            self.path_squared_errors = []
        except Exception:
            self.close()
            raise

    def _marker(self, pos):
        if self.cfg.task == "pick-place":
            visual = self.client.createVisualShape(self.p.GEOM_BOX, halfExtents=[.055, .055, .001],
                                                   rgbaColor=[.15, .7, .3, .45])
            return self.client.createMultiBody(0., -1, visual, [pos[0], pos[1], .001])
        visual = self.client.createVisualShape(self.p.GEOM_SPHERE, radius=.025,
                                               rgbaColor=[.15, .7, .3, .45])
        return self.client.createMultiBody(0., -1, visual, pos)

    def _motors(self):
        self.client.setJointMotorControlArray(
            self.robot, self.joints, self.p.POSITION_CONTROL, targetPositions=self.motor_targets.tolist(),
            forces=[120.] * 7 + [35., 35.], positionGains=[.3] * 9, velocityGains=[1.] * 9)

    def state(self):
        link = self.client.getLinkState(self.robot, self.ee, computeForwardKinematics=True)
        joints = self.client.getJointStates(self.robot, self.joints)
        op, oq = self.client.getBasePositionAndOrientation(self.cube)
        q = np.array([s[0] for s in joints])
        return RobotState(np.asarray(link[4]), np.asarray(link[5]), q,
                          np.array([s[1] for s in joints]), float(q[-2:].sum()), np.asarray(op), np.asarray(oq))

    def _near_links(self, a, b):
        ancestors, node = {a: 0}, a
        for depth in (1, 2):
            node = self.parents.get(node, -1)
            ancestors[node] = depth
        node = b
        for depth in (0, 1, 2):
            if node in ancestors and depth + ancestors[node] <= 2:
                return True
            node = self.parents.get(node, -1)
        return False

    def _ik(self, command, state):
        self.reason = "ok"
        try:
            target, quat = array(command.position, (3,)), array(command.quaternion, (4,))
            if np.any(target < np.array(self.cfg.workspace_min) - 1e-6) or np.any(
                    target > np.array(self.cfg.workspace_max) + 1e-6):
                raise ValueError("outside_workspace")
            for j, q in zip(self.joints, state.joints):
                self.kin.resetJointState(self.kin_robot, j, float(q))
            solution = self.kin.calculateInverseKinematics(
                self.kin_robot, self.ee, target, quat, lowerLimits=self.lower.tolist(),
                upperLimits=self.upper.tolist(), jointRanges=(self.upper - self.lower).tolist(),
                restPoses=REST, jointDamping=[.05] * 9, maxNumIterations=100, residualThreshold=1e-5)
            q = np.r_[np.asarray(solution[:7]), [np.clip(command.grip / 2., 0., .04)] * 2]
            if not np.isfinite(q).all() or np.any(q < self.lower - .02) or np.any(q > self.upper + .02):
                raise ValueError("joint_limit")
            q = np.clip(q, self.lower, self.upper)
            for j, value in zip(self.joints, q):
                self.kin.resetJointState(self.kin_robot, j, float(value))
            fk = self.kin.getLinkState(self.kin_robot, self.ee, computeForwardKinematics=True)
            pos_error = np.linalg.norm(np.asarray(fk[4]) - target)
            rot_error = (Rotation.from_quat(fk[5]) * Rotation.from_quat(quat).inv()).magnitude()
            if pos_error > .035 or rot_error > .3:
                raise ValueError("ik_residual")
            self.kin.performCollisionDetection()
            for contact in self.kin.getContactPoints(self.kin_robot, self.kin_floor):
                if contact[3] >= 1 and contact[8] < -.001:
                    raise ValueError("arm_floor_collision")
            for contact in self.kin.getContactPoints(self.kin_robot, self.kin_robot):
                if not self._near_links(contact[3], contact[4]) and contact[8] < -.002:
                    raise ValueError("self_collision")
            return q
        except (ValueError, self.p.error) as exc:
            self.reason = str(exc)
            return None

    def step(self, command):
        state = self.state()
        target = self._ik(command, state) if command.active else None
        if command.active and target is None:
            self.rejections += 1
        if target is not None:
            dq = self.cfg.max_joint_speed / self.cfg.hz
            self.motor_targets[:7] = state.joints[:7] + np.clip(target[:7] - state.joints[:7], -dq, dq)
            self.motor_targets[7:] = target[7:]
            self.holding = False
        elif not self.holding:
            self.motor_targets[:7] = state.joints[:7]
            self.holding = True
        self._motors()
        for _ in range(240 // self.cfg.hz):
            self.client.stepSimulation()
        self.ticks += 1
        self.active_ticks += int(command.active)
        self.time, self.active_time = self.ticks / self.cfg.hz, self.active_ticks / self.cfg.hz
        after = self.state()
        self._task(after, command.active)
        return after

    def goal(self, t=None):
        if self.cfg.task == "pick-place":
            return self.place_goal.copy()
        if self.cfg.task == "reach":
            return self.reach_goal.copy()
        t = self.active_time if t is None else t
        angle = max(0., min(t - 3., 16.)) * 2. * np.pi / 16.
        return np.array([.5, .10 * np.cos(angle), .30 + .08 * np.sin(angle)])

    def _task(self, state, active):
        if not active:
            return
        if self.cfg.task == "reach":
            condition = np.linalg.norm(state.position - self.reach_goal) < .025
        elif self.cfg.task == "pick-place":
            self.lifted |= state.object_position[2] > .1
            condition = (self.lifted and np.linalg.norm(state.object_position[:2] - self.place_goal[:2]) < .055
                         and .014 < state.object_position[2] < .06 and state.grip > .055
                         and np.linalg.norm(state.position - state.object_position) > .075)
        else:
            goal = self.goal()
            self.client.resetBasePositionAndOrientation(self.goal_body, goal, [0., 0., 0., 1.])
            if 3. <= self.active_time <= 19.:
                self.path_squared_errors.append(float(np.sum((state.position - goal) ** 2)))
            condition = (self.active_time >= 19. and bool(self.path_squared_errors)
                         and np.sqrt(np.mean(self.path_squared_errors)) < .035)
        self.dwell = self.dwell + 1. / self.cfg.hz if condition else 0.
        if self.dwell >= .5 and not self.success:
            self.success, self.success_time = True, self.active_time

    def task_summary(self):
        return {"task": self.cfg.task, "success": bool(self.success), "success_active_time_s": self.success_time,
                "cube_lifted": bool(self.lifted), "ik_rejections": self.rejections,
                "simulation_seconds": self.time, "active_seconds": self.active_time,
                "path_rmse_m": float(np.sqrt(np.mean(self.path_squared_errors)))
                if self.path_squared_errors else None}

    def restore_visual_state(self, state):
        """Kinematic presentation ONLY. Never used by simulation or success tests."""
        for j, q in zip(self.joints, state["joints"]):
            self.client.resetJointState(self.robot, j, q)
        self.client.resetBasePositionAndOrientation(self.cube, state["object_position"], state["object_quaternion"])

    def render(self, width=640, height=480, top=False):
        view = self.client.computeViewMatrixFromYawPitchRoll(
            [.45, 0., .15], 1.15, 65 if not top else 90, -35 if not top else -89, 0, 2)
        projection = self.client.computeProjectionMatrixFOV(50, width / height, .02, 5.)
        image = self.client.getCameraImage(width, height, view, projection,
                                           renderer=self.p.ER_TINY_RENDERER)[2]
        return np.asarray(image, dtype=np.uint8).reshape(height, width, 4)[:, :, :3][:, :, ::-1].copy()

    def close(self):
        if not self.closed:
            self.closed = True
            for client in (self.client, self.kin):
                if client.isConnected():
                    client.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
