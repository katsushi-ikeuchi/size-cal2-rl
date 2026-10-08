import argparse
from collections import deque
import math
import os
import sys
import time

import pybullet as p
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(SCRIPT_DIR, "panda_hx5.usd")

def create_isaac_env2(initial_curriculum_scale=0.10, fix_box_pos=False, simulation_app=None, headless=True, base_box_size=None):
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces

    import omni.isaac.core.utils.prims as prim_utils
    import omni.isaac.core.utils.stage as stage_utils
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
    from omni.isaac.core.utils.types import ArticulationAction


    try:
        from isaacsim.asset.importer.urdf import _urdf
        import omni.kit.commands
    except ImportError:
        try:
            from omni.isaac.urdf import _urdf
            import omni.kit.commands
        except ImportError:
            pass

    URDF_PATH = "/home/ikeuchi/flower/calvin_env/data/panda_hx5/panda_hx5_right.urdf"
    ARM_JOINT_NAMES = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7"
    ]
    FINGER_JOINT_NAMES = [f"finger_r_joint{i}" for i in range(1, 21)]

    class RobotisGraspSizeCal2IsaacEnv(gym.Env):
        """
        [size_cal2.py] Isaac Sim 5.1.0 用 インクリメンタル制御 & 力覚(トルク)観測 & オラクル力つり合い報酬環境
        """

        def __init__(self, headless=True):
            super(RobotisGraspSizeCal2IsaacEnv, self).__init__()
            self.headless = headless
            # アクション空間 (2次元: [a_thumb, a_finger] in [-1.0, 1.0])
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            # 観測空間 (64次元: 関節15 + 箱位置3 + 指姿勢20 + 箱寸法3 + 生トルク20 + 前回アクション2 + ステップ進行度1)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)

            self.previous_action = np.zeros(2, dtype=np.float32)

            self.fix_box_pos = fix_box_pos
            self.simulation_app = simulation_app
            self.stable_lift_count = 0
            
            # 釣り合いと持ち上げフラグ
            self.is_ready_to_lift = False
            self.grasp_stable_count = 0
            self.lift_start_step = 0

            self.curriculum_scale = float(initial_curriculum_scale)
            if base_box_size is not None:
                self.base_box_size = np.array(base_box_size)
            else:
                self.base_box_size = np.array([0.025, 0.025, 0.025])  # 基本5cm (4cm〜6cmの範囲)
            self.max_var_size = np.array([0.005, 0.005, 0.005])  # 最大±1cmの変動幅
            self.current_box_size = list(self.base_box_size)
            self.current_box_pos = [0.635, 0.0, 0.45 + self.base_box_size[2]]

            self.world = World(stage_units_in_meters=1.0)
            self.world.scene.add_default_ground_plane()

            self.table = FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=np.array([0.6, 0.0, 0.225]),
                scale=np.array([0.60, 0.60, 0.45]),
                color=np.array([0.55, 0.40, 0.25]),
            )

            self.robot_prim_path = "/World/PandaHX5"
            self._import_robot_urdf(URDF_PATH)
            self.robot = Articulation(prim_path=self.robot_prim_path, name="robot")
            self.world.scene.add(self.robot)

            self.box_prim_path = "/World/Box"
            self.box = DynamicCuboid(
                prim_path=self.box_prim_path,
                name="grasp_box",
                position=np.array(self.current_box_pos),
                scale=np.array(self.base_box_size) * 2.0,
                color=np.array([0.1, 0.7, 0.9]),
                mass=0.01,  # 10gに変更
            )
            self.world.scene.add(self.box)

            self.world.reset()
            self.world.step(render=not self.headless)
            self.robot.initialize()
            
            from omni.isaac.core.prims import RigidPrim
            from omni.isaac.core.prims import XFormPrim
            self.tcp_prim = RigidPrim(prim_path="/World/PandaHX5/tcp", name="tcp_prim")
            
            # 指先や手首などのPrimを取得 (強化学習の報酬や観測用)
            self.thumb_prim = XFormPrim(prim_path="/World/PandaHX5/finger_end_r_link1", name="thumb_prim")
            self.index_prim = XFormPrim(prim_path="/World/PandaHX5/finger_r_link8", name="index_prim")
            self.middle_prim = XFormPrim(prim_path="/World/PandaHX5/finger_end_r_link3", name="middle_prim")
            self.ring_prim = XFormPrim(prim_path="/World/PandaHX5/finger_r_link16", name="ring_prim")
            self.pinky_prim = XFormPrim(prim_path="/World/PandaHX5/finger_r_link20", name="pinky_prim")
            
            self.tcp_prim.initialize()
            self.thumb_prim.initialize()
            self.index_prim.initialize()
            self.middle_prim.initialize()
            self.ring_prim.initialize()
            self.pinky_prim.initialize()


            all_dofs = set(self.robot.dof_names) if self.robot.dof_names else set()
            self.arm_joint_indices = [self.robot.get_dof_index(name) for name in ARM_JOINT_NAMES if name in all_dofs]
            self.finger_joint_indices = [self.robot.get_dof_index(name) for name in FINGER_JOINT_NAMES if name in all_dofs]

            self.step_counter = 0
            self.max_steps = 50

            self.arm_grasp_q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.57, 0.785])
            self.base_finger_q = np.zeros(len(self.finger_joint_indices))
            self.current_finger_q = np.zeros(len(self.finger_joint_indices))

            # [NEW] PyBullet IK setup
            self.pb_cid = p.connect(p.DIRECT)
            self.pb_robot = p.loadURDF(URDF_PATH, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, physicsClientId=self.pb_cid)
            
            # Get link indices for IK
            self.pb_link_indices = {}
            self.pb_joint_indices = {}
            for i in range(p.getNumJoints(self.pb_robot, physicsClientId=self.pb_cid)):
                info = p.getJointInfo(self.pb_robot, i, physicsClientId=self.pb_cid)
                j_name = info[1].decode("utf-8")
                l_name = info[12].decode("utf-8")
                self.pb_link_indices[l_name] = i
                self.pb_joint_indices[j_name] = i

            # Baseline arm joint poses
            self.pb_rest_poses = [-0.01664124, 0.07485258, -0.01176343, -1.42680791, -0.01028490, 1.16893920, -2.39960431]
            for i in range(7):
                p.resetJointState(self.pb_robot, i, self.pb_rest_poses[i], physicsClientId=self.pb_cid)

        def _import_robot_urdf(self, urdf_path):
            if not os.path.exists(USD_PATH):
                import_config = _urdf.ImportConfig()
                import_config.merge_fixed_joints = False
                import_config.convex_decomp = False
                import_config.import_inertia_tensor = True
                import_config.fix_base = True
                import_config.make_default_prim = True
                omni.kit.commands.execute(
                    "URDFParseAndImportFile",
                    urdf_path=urdf_path,
                    import_config=import_config,
                    dest_path=USD_PATH,
                )
            stage_utils.add_reference_to_stage(usd_path=USD_PATH, prim_path=self.robot_prim_path)

        def set_curriculum_scale(self, scale: float):
            self.curriculum_scale = max(0.05, min(1.0, scale))

        def get_curriculum_scale(self) -> float:
            return self.curriculum_scale

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)

            # 1. 箱サイズの更新
            size_var = self.max_var_size * self.curriculum_scale
            rand_var = np.random.uniform(-size_var, size_var)
            self.current_box_size = list(np.clip(self.base_box_size + rand_var, 0.020, 0.030))

            # 物理テンソル無効化とスケール初期化のジレンマを完全に回避するため、
            # シミュレーションを停止 → スケール変更 → リセット（再起動） の順で処理する
            self.world.stop()
            self.box.set_local_scale(np.array(self.current_box_size) * 2.0)
            self.world.reset()

            # 2. 箱位置の更新

            if self.fix_box_pos:
                random_x = 0.635
                random_y = 0.0
            else:
                pos_var = 0.02 * self.curriculum_scale
                random_x = 0.635 + np.random.uniform(-pos_var, pos_var)
                random_y = 0.0 + np.random.uniform(-pos_var, pos_var)

            box_z = 0.45 + self.current_box_size[2]
            self.current_box_pos = [random_x, random_y, box_z]
            self.box.set_world_pose(position=np.array(self.current_box_pos), orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            self.box.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            self.box.set_angular_velocity(np.array([0.0, 0.0, 0.0]))


            # ★ 箱サイズに応じた指の初期プレオープン姿勢 (PyBullet size_cal4と完全完全一致)
            dx = self.current_box_size[0]
            dy = self.current_box_size[1]
            self.base_finger_q = np.zeros(len(self.finger_joint_indices), dtype=np.float32)
            
            base_clearance = 0.30
            x_offset = max(0.0, (dx - 0.015) * 1.5)
            y_offset = (dy - 0.04) * 1.5
            
            # Apply to all fingers first
            self.base_finger_q[2::4] += (base_clearance + x_offset)
            self.base_finger_q[3::4] += (base_clearance + x_offset)
            self.base_finger_q[1::4] += y_offset * 0.5
            
            # 親指(Thumb)の特別調整
            self.base_finger_q[0] = 0.175  # 第一関節: ロール(10度ひねって中指と対向させる)
            self.base_finger_q[1] = -1.745 # 第二関節: 屈曲(-100度)
            self.base_finger_q[2] = 0.175  # 第三関節: 10度
            self.base_finger_q[3] = 0.0    # 第四関節: 0度

            self.current_finger_q = np.array(self.base_finger_q)

            # ★ 把持線の中点が物体の中心を通るように、PyBulletを用いた正確なIK計算を行う
            
            # 1. PyBulletのロボットを基準姿勢（アーム）と計算したプレシェイプ（指）にセットする
            for i in range(7):
                p.resetJointState(self.pb_robot, self.pb_joint_indices[f"panda_joint{i+1}"], self.pb_rest_poses[i], physicsClientId=self.pb_cid)
            
            for i in range(20):
                j_name = f"finger_r_joint{i+1}"
                if j_name in self.pb_joint_indices:
                    p.resetJointState(self.pb_robot, self.pb_joint_indices[j_name], self.base_finger_q[i], physicsClientId=self.pb_cid)

            # --- Iterative PyBullet Calibration ---
            # URDF(PyBullet)とUSD(Isaac Sim)のわずかなリンク長/ベース座標のズレを補正するため、
            # Isaac Sim上で数ステップ進めて実際の把持線中点を計測し、PyBulletの目標座標を微修正する。
            
            # 1. 基準となる手首姿勢と、手首から把持線中点への相対ベクトルを計算（一度だけ）
            for i in range(7):
                p.resetJointState(self.pb_robot, self.pb_joint_indices[f"panda_joint{i+1}"], self.pb_rest_poses[i], physicsClientId=self.pb_cid)
            for i in range(20):
                j_name = f"finger_r_joint{i+1}"
                if j_name in self.pb_joint_indices:
                    p.resetJointState(self.pb_robot, self.pb_joint_indices[j_name], self.base_finger_q[i], physicsClientId=self.pb_cid)
                    
            pb_tcp_state = p.getLinkState(self.pb_robot, self.pb_link_indices["tcp"], physicsClientId=self.pb_cid)
            base_tcp_pos = np.array(pb_tcp_state[0])
            base_tcp_quat = np.array(pb_tcp_state[1])
            
            pb_thumb = np.array(p.getLinkState(self.pb_robot, self.pb_link_indices["finger_end_r_link1"], physicsClientId=self.pb_cid)[0])
            pb_middle = np.array(p.getLinkState(self.pb_robot, self.pb_link_indices["finger_end_r_link3"], physicsClientId=self.pb_cid)[0])
            base_midpoint = (pb_thumb + pb_middle) / 2.0
            
            base_midpoint_offset = base_midpoint - base_tcp_pos
            
            box_center_pos = np.array([self.current_box_pos[0], self.current_box_pos[1], self.current_box_pos[2]])
            # 手の姿勢が斜めのため、親指が机にめり込まないようオフセットを設定
            z_offset = 0.015
            target_midpoint = box_center_pos + np.array([0.0, 0.0, z_offset])
            pb_target_midpoint = np.copy(target_midpoint)
            
            self.arm_grasp_q = np.zeros(7, dtype=np.float32)
            
            for it in range(5): # 最大5回の補正ループ
                # 常に同じ相対ベクトルと姿勢を使用してIK目標TCP座標を計算
                pb_target_tcp = pb_target_midpoint - base_midpoint_offset
                
                q_ik = p.calculateInverseKinematics(
                    self.pb_robot,
                    self.pb_link_indices["tcp"],
                    pb_target_tcp,
                    targetOrientation=base_tcp_quat, # 現在の手首の向きを維持
                    maxNumIterations=200,
                    residualThreshold=1e-5,
                    physicsClientId=self.pb_cid
                )
                
                for i in range(7):
                    self.arm_grasp_q[i] = q_ik[self.pb_joint_indices[f"panda_joint{i+1}"]]
                    
                # Isaac Simのロボットにアームと指の角度をPDターゲットとして設定し、重力によるたわみ（Gravity Sag）を含めて評価する
                self.robot.set_joint_positions(self.arm_grasp_q, self.arm_joint_indices)
                self.robot.set_joint_positions(self.base_finger_q, self.finger_joint_indices)
                
                self.robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=self.arm_grasp_q, joint_indices=self.arm_joint_indices)
                )
                self.robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=self.base_finger_q, joint_indices=self.finger_joint_indices)
                )
                
                # PD制御が安定し、重力たわみが反映されるまで10ステップ進める
                for _ in range(10):
                    self.box.set_world_pose(position=self.current_box_pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
                    self.box.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                    self.box.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                    self.world.step(render=False)
                
                # 重力たわみを含んだ最終的な把持線中点を計測
                isaac_thumb, _ = self.thumb_prim.get_world_pose()
                isaac_middle, _ = self.middle_prim.get_world_pose()
                current_midpoint = (isaac_thumb + isaac_middle) / 2.0
                
                error = target_midpoint - current_midpoint
                err_norm = np.linalg.norm(error)
                
                # 誤差が2mm未満になればループ終了
                if err_norm < 0.002:
                    break
                    
                # 誤差分だけPyBullet側の目標座標をシフト（学習率0.8）
                pb_target_midpoint = pb_target_midpoint + error * 0.8

            self.robot.set_joint_positions(self.arm_grasp_q, self.arm_joint_indices)
            self.robot.set_joint_positions(self.base_finger_q, self.finger_joint_indices)

            self.robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=self.arm_grasp_q, joint_indices=self.arm_joint_indices)
            )
            self.robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=self.base_finger_q, joint_indices=self.finger_joint_indices)
            )

            for _ in range(10):
                self.world.step(render=not self.headless)

            self.step_counter = 0
            self.stable_lift_count = 0
            self.is_ready_to_lift = False
            self.grasp_stable_count = 0
            self.lift_start_step = 0
            
            self.previous_action = np.zeros(2, dtype=np.float32)
            return self._get_obs(), {}

        def _get_obs(self):
            # ステップを進めて物理を反映
            self.world.step(render=False)


            obs = []
            robot_dof_pos = self.robot.get_joint_positions()
            # 1. 主要関節 (15)
            obs.extend(robot_dof_pos[:15] if len(robot_dof_pos) >= 15 else np.pad(robot_dof_pos, (0, 15 - len(robot_dof_pos))))

            # 2. 箱の3次元位置 (3)
            box_pos, _ = self.box.get_world_pose()
            obs.extend(box_pos)

            # 3. 全指関節の現在の角度 (20)
            finger_pos = robot_dof_pos[7:27] if len(robot_dof_pos) >= 27 else np.pad(robot_dof_pos[7:], (0, 20 - len(robot_dof_pos[7:])))
            obs.extend(finger_pos)

            # 4. 箱のサイズ寸法 (3)
            obs.extend(self.current_box_size)

            # 5. [NEW] 生の指モータートルク / 電流相当値 (20)
            measured_efforts = self.robot.get_measured_joint_efforts()
            finger_efforts = measured_efforts[7:27] if len(measured_efforts) >= 27 else np.pad(measured_efforts[7:], (0, 20 - len(measured_efforts[7:])))
            obs.extend(finger_efforts)

            # 6. [NEW] 前回のアクション (2)
            obs.extend(self.previous_action)

            # 7. [NEW] ステップ進行度 (1)
            step_progress = self.step_counter / self.max_steps
            obs.append(step_progress)

            return np.array(obs[:64], dtype=np.float32)

        def step(self, action):
            self.step_counter += 1

            action_clipped = np.clip(action, -1.0, 1.0)
            self.previous_action = np.copy(action_clipped)
            a_thumb = action_clipped[0]
            a_finger = action_clipped[1]

            delta_thumb = a_thumb * 0.02
            delta_finger = a_finger * 0.03

            for i in range(min(5, len(self.finger_joint_indices) // 4)):
                val = delta_thumb if i == 0 else delta_finger
                idx = i * 4
                if idx + 3 < len(self.current_finger_q):
                    if i == 0:
                        # 親指: 第3・第4のみ更新し、クリップする
                        self.current_finger_q[idx + 2] = np.clip(self.current_finger_q[idx + 2] + val, -0.2, 0.65)
                        self.current_finger_q[idx + 3] = np.clip(self.current_finger_q[idx + 3] + val, -0.2, 0.65)
                    else:
                        # 他の指: 従来通り
                        self.current_finger_q[idx + 1] = np.clip(self.current_finger_q[idx + 1] + val, -0.2, 0.65)
                        self.current_finger_q[idx + 2] = np.clip(self.current_finger_q[idx + 2] + val, -0.2, 0.65)
                        self.current_finger_q[idx + 3] = np.clip(self.current_finger_q[idx + 3] + val, -0.2, 0.65)

            if len(self.finger_joint_indices) > 0:
                self.robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=self.current_finger_q, joint_indices=self.finger_joint_indices)
                )

            # 5 sub-steps per environment step for physics PD settling
            for _ in range(5):
                self.world.step(render=not self.headless)

            # --- 距離と状態の計算 ---
            box_pos, box_quat = self.box.get_world_pose()
            lift_height = box_pos[2] - self.current_box_pos[2]
            
            # 各指先の座標を取得
            thumb_pos, _ = self.thumb_prim.get_world_pose()
            index_pos, _ = self.index_prim.get_world_pose()
            middle_pos, _ = self.middle_prim.get_world_pose()
            ring_pos, _ = self.ring_prim.get_world_pose()
            pinky_pos, _ = self.pinky_prim.get_world_pose()
            
            # 箱の中心までの距離（各指）
            dist_thumb = np.linalg.norm(thumb_pos - box_pos)
            dist_index = np.linalg.norm(index_pos - box_pos)
            dist_middle = np.linalg.norm(middle_pos - box_pos)
            dist_ring = np.linalg.norm(ring_pos - box_pos)
            dist_pinky = np.linalg.norm(pinky_pos - box_pos)
            
            # 5本の指の距離の平均（これが小さいほど、全指が箱の中心に集まっている＝内向きに閉じている）
            avg_finger_dist = (dist_thumb + dist_index + dist_middle + dist_ring + dist_pinky) / 5.0
            
            # tcp_posは全体の中心として計算（距離チェック用）
            tcp_pos = (thumb_pos + index_pos + middle_pos + ring_pos + pinky_pos) / 5.0
            dist_to_box = np.linalg.norm(tcp_pos - box_pos)

            # --- 力の釣り合い計算 ---
            actual_finger_q = self.robot.get_joint_positions()[7:27]
            measured_efforts = self.robot.get_measured_joint_efforts()
            finger_efforts = measured_efforts[7:27] if len(measured_efforts) >= 27 else np.zeros(20)
            
            # モータートルクの絶対値の合計で「掴んでいる力」を判定する
            total_finger_effort = np.sum(np.abs(finger_efforts))

            # --- 指の速度チェック（フライング防止） ---
            actual_finger_vel = self.robot.get_joint_velocities()[7:27]
            # Isaac Simでは衝突時に速度が跳ねてしまうため、完全静止を条件に入れるとクリア不可能になっていました
            is_stopped = np.max(np.abs(actual_finger_vel)) < 0.5 if len(actual_finger_vel) > 0 else False

            # 指がある程度閉じており、かつトルクが発生している場合のみ「掴んだ」と判定
            # (is_stopped は厳しすぎるため条件から外す)
            # シミュレータの指関節のトルク上限（約0.6）を考慮し、閾値を 0.45 に設定
            is_force_balanced = (total_finger_effort > 0.45) and (avg_finger_dist < 0.09) and (self.step_counter > 15)

            if not self.headless:
                if self.step_counter % 10 == 0 or self.is_ready_to_lift:
                    print(f"[Step {self.step_counter}] is_ready={self.is_ready_to_lift}, is_balanced={is_force_balanced}")
                    print(f"  Effort={total_finger_effort:.3f}, Dist={avg_finger_dist:.3f}, Box_Z={box_pos[2]:.4f}")

            if is_force_balanced and not self.is_ready_to_lift:
                self.grasp_stable_count += 1
            elif not self.is_ready_to_lift:
                self.grasp_stable_count = 0

            if self.grasp_stable_count >= 5 and not self.is_ready_to_lift:
                self.is_ready_to_lift = True
                self.lift_start_step = self.step_counter

            # --- 持ち上げ ---
            if self.is_ready_to_lift:
                target_lift_q = np.array(self.arm_grasp_q)
                lift_progress = min(20.0, self.step_counter - self.lift_start_step)
                target_lift_q[1] -= 0.3 * (lift_progress / 20.0)
                self.robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=target_lift_q, joint_indices=self.arm_joint_indices)
                )
                
            # --- 報酬計算 ---
            reward = 0.0
            
            # 1. 指先が箱の中心に集まる（内向きに閉じる）ことへの報酬
            # avg_finger_dist が 0.10(10cm) から 0.0(0cm) に近づくほど報酬増
            dist_reward = max(0.0, 0.10 - avg_finger_dist) * 10.0
            reward += dist_reward
            
            # 2. 掴む力（トルク）に対するボーナス
            # 箱に触れている（avg_finger_dist < 0.12）間に強く握り込むほど報酬を与える
            if avg_finger_dist < 0.12 and not self.is_ready_to_lift:
                effort_bonus = np.clip(total_finger_effort / 2.0, 0.0, 1.0) * 1.5
                reward += effort_bonus

            # 3. 釣り合い（持ち上げ準備完了）ボーナス
            if is_force_balanced:
                reward += 0.5
            angle_diff = 2.0 * math.acos(np.clip(abs(box_quat[0]), 0.0, 1.0))
            reward -= angle_diff * 0.25

            done = False
            info = {}

            if self.is_ready_to_lift:
                if lift_height > 0.01:
                    reward += lift_height * 40.0

                if lift_height > 0.025:
                    self.stable_lift_count += 1
                else:
                    self.stable_lift_count = 0

                if self.stable_lift_count >= 10:
                    remaining_steps = self.max_steps - self.step_counter
                    reward += 30.0 + remaining_steps * 1.0
                    info["is_success"] = True
                    done = True

            if self.step_counter >= self.max_steps and not done:
                done = True
                if "is_success" not in info:
                    info["is_success"] = False

            obs = self._get_obs()
            truncated = False
            return obs, reward, done, truncated, info

        def close(self):
            if self.simulation_app is not None:
                self.simulation_app.close()
            try:
                p.disconnect(self.pb_cid)
            except Exception:
                pass

    return RobotisGraspSizeCal2IsaacEnv(headless=headless)


def make_worker_env2(rank, gpu_id, initial_scale, fix_pos, display=False):
    def _init():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        from omni.isaac.kit import SimulationApp
        app = SimulationApp({"headless": not display, "active_gpu": 0, "physics_gpu": 0})
        env = create_isaac_env2(
            initial_curriculum_scale=initial_scale,
            fix_box_pos=fix_pos,
            simulation_app=app,
        )
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# 2. 自動カリキュラム管理 Callback (双方向 昇格・降格対応)
# ─────────────────────────────────────────────────────────────────────────────
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class CurriculumCallback2(BaseCallback):
    def __init__(
        self,
        eval_env,
        window_size=100,
        success_threshold=0.80,
        downgrade_threshold=0.50,
        step_increment=0.05,
        save_dir=None,
        verbose=1,
    ):
        super(CurriculumCallback2, self).__init__(verbose)
        self.eval_env = eval_env
        self.window_size = window_size
        self.success_threshold = success_threshold
        self.downgrade_threshold = downgrade_threshold
        self.step_increment = step_increment
        self.save_dir = save_dir
        self.recent_successes = deque(maxlen=window_size)
        self.last_scale_update_step = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "is_success" in info:
                is_succ = 1.0 if info["is_success"] else 0.0
                self.recent_successes.append(is_succ)
            elif "episode" in info and "is_success" in info.get("episode", {}):
                is_succ = 1.0 if info["episode"]["is_success"] else 0.0
                self.recent_successes.append(is_succ)

        current_scale = 0.10
        if hasattr(self.training_env, "env_method"):
            scales = self.training_env.env_method("get_curriculum_scale")
            if scales:
                current_scale = scales[0]

        if len(self.recent_successes) >= 10:
            import numpy as np
            success_rate = float(np.mean(self.recent_successes))

            if self.logger:
                self.logger.record("curriculum/success_rate_recent", success_rate)
                self.logger.record("curriculum/scale", current_scale)

            if self.num_timesteps % 1000 == 0:
                print(
                    f"  [Isaac Sim size_cal2] Step: {self.num_timesteps} | "
                    f"Scale: {current_scale:.2f} | "
                    f"Recent Success Rate: {success_rate * 100:.1f}%"
                )

            # 1) カリキュラム昇格 (Upgrade)
            if (
                success_rate >= self.success_threshold
                and len(self.recent_successes) >= 30
                and (self.num_timesteps - self.last_scale_update_step) > 2000
            ):
                if current_scale < 1.0:
                    new_scale = min(1.0, current_scale + self.step_increment)
                    if hasattr(self.training_env, "env_method"):
                        self.training_env.env_method("set_curriculum_scale", new_scale)
                    self.last_scale_update_step = self.num_timesteps

                    print(
                        f"\n🎉 [Automatic Curriculum Upgrade] Step {self.num_timesteps}: "
                        f"Recent Success Rate = {success_rate*100:.1f}% (>= {self.success_threshold*100:.0f}%). "
                        f"Upgrading size & position scale: {current_scale:.2f} ➔ {new_scale:.2f}\n"
                    )
                    self.recent_successes.clear()

            # 2) カリキュラム降格 (Downgrade: 退避機能)
            elif (
                success_rate < self.downgrade_threshold
                and len(self.recent_successes) >= 30
                and (self.num_timesteps - self.last_scale_update_step) > 2000
            ):
                if current_scale > 0.05:
                    new_scale = max(0.05, current_scale - self.step_increment)
                    if hasattr(self.training_env, "env_method"):
                        self.training_env.env_method("set_curriculum_scale", new_scale)
                    self.last_scale_update_step = self.num_timesteps

                    print(
                        f"\n🔻 [Automatic Curriculum Downgrade] Step {self.num_timesteps}: "
                        f"Recent Success Rate = {success_rate*100:.1f}% (< {self.downgrade_threshold*100:.0f}%). "
                        f"Downgrading size & position scale: {current_scale:.2f} ➔ {new_scale:.2f}\n"
                    )
                    self.recent_successes.clear()

        return True


# ─────────────────────────────────────────────────────────────────────────────
# 3. メイン実行エントリーポイント
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Robotis Grasp SAC size_cal2.py (Isaac Sim 5.1.0)")
    parser.add_argument("--timesteps", type=int, default=10000000, help="総学習ステップ数 (デフォルト: 10000000)")
    parser.add_argument("--initial-scale", type=float, default=0.10, help="初期カリキュラムスケール (デフォルト: 0.10)")
    parser.add_argument("--fix-pos", action="store_true", help="位置ランダム化を無効にして中央固定にする")
    parser.add_argument("--debug", action="store_true", help="GUIを表示して保存済みモデルをテストする")
    parser.add_argument("--gui", action="store_true", help="学習中もGUIを表示する（動作確認用）")
    parser.add_argument("--resume", action="store_true", help="最新のチェックポイントから学習を再開する")
    args = parser.parse_args()

    show_gui = args.debug or args.gui

    from omni.isaac.kit import SimulationApp
    app = SimulationApp({"headless": not show_gui, "active_gpu": 0, "physics_gpu": 0})

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(SCRIPT_DIR, "log2")
    SAVE_DIR = os.path.join(SCRIPT_DIR, "checkpoints2")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n=== Starting Robotis Grasp SAC size_cal2.py in Isaac Sim ===")
    print(f"Log Dir: {LOG_DIR}")
    print(f"Save Dir (Checkpoints): {SAVE_DIR}")
    print("Features: Incremental Delta Control + 61-Dim Torque Obs + PhysX Oracle Force Balance Reward\n")

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    raw_env = create_isaac_env2(
        initial_curriculum_scale=args.initial_scale,
        fix_box_pos=args.fix_pos,
        simulation_app=app,
        headless=not show_gui
    )
    if show_gui:
        import numpy as np
        from omni.isaac.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([0.635, 0.5, 0.48]), target=np.array([0.635, 0.0, 0.48]))

        from pxr import UsdLux, Gf
        stage = raw_env.world.stage
        extra_light = UsdLux.DistantLight.Define(stage, '/World/ExtraLight')
        extra_light.CreateIntensityAttr(2000.0)
        extra_light.CreateAngleAttr(1.0)
        extra_light.AddOrientOp().Set(Gf.Quatf(0.853, 0.353, 0.353, 0.146))

    vec_env = DummyVecEnv([lambda: raw_env])

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

    if args.debug:
        print("\n=== DEBUG MODE: Evaluation ===")
        import glob
        list_of_files = glob.glob(os.path.join(SAVE_DIR, '*.zip'))
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getmtime)
            print(f"Loading latest checkpoint: {latest_file}")
            model = SAC.load(latest_file, env=vec_env, device="cuda:0")
            
            obs = vec_env.reset()
            print("Starting evaluation loop. Press Ctrl+C to stop.")
            try:
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = vec_env.step(action)
            except KeyboardInterrupt:
                print("\nEvaluation stopped by user.")
        else:
            print(f"No checkpoints found in {SAVE_DIR}. Cannot evaluate.")
        
        raw_env.close()
        return

    if args.resume:
        import glob
        import re
        list_of_files = glob.glob(os.path.join(SAVE_DIR, '*.zip'))
        if list_of_files:
            def get_step_from_filename(filename):
                match = re.search(r'(\d+)_steps\.zip$', filename)
                return int(match.group(1)) if match else -1
            
            latest_file = max(list_of_files, key=get_step_from_filename)
            print(f"\n=== RESUMING from checkpoint: {latest_file} ===")
            model = SAC.load(latest_file, env=vec_env, device="cuda:0", tensorboard_log=LOG_DIR)
        else:
            print("\nNo checkpoints found to resume from. Exiting.")
            raw_env.close()
            return
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=50000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        verbose=1,
        device="cuda:0",
    )

    curriculum_cb = CurriculumCallback2(
        eval_env=vec_env,
        window_size=100,
        success_threshold=0.80,
        downgrade_threshold=0.50,
        step_increment=0.05,
        save_dir=SAVE_DIR,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=1000,
        save_path=SAVE_DIR,
        name_prefix="robotis_grasp_sac_isaac2_4gpu",
    )

    print(f"★ 学習開始: 総ステップ数 {args.timesteps:,} ...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[curriculum_cb, checkpoint_cb],
        progress_bar=False,
        reset_num_timesteps=not args.resume,
    )

    final_model_path = os.path.join(SAVE_DIR, "robotis_grasp_sac_isaac2_4gpu_final.zip")
    model.save(final_model_path)
    print(f"\n🎉 学習完了！ 最終モデル保存先: {final_model_path}")

    raw_env.close()


if __name__ == "__main__":
    main()
