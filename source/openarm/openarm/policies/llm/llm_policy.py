from isaaclab_arena.policy.policy_base import PolicyBase
import omni.client
from scene_synthesizer.assets import USDAsset
import json
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
import torch
import curobo
import isaaclab.utils.math as PoseUtils

class LLMPolicy(PolicyBase):
    def __init__(self, scene, task_description):
        super().__init__()
        scene_description = {}
        for asset_name in scene.assets.keys():
            scene_description[asset_name] = {}
            if scene.assets[asset_name].usd_path.startswith('https://'):
                omni.client.copy(scene.assets[asset_name].usd_path, "/tmp/downloaded_usd.usd")
                usd_asset = USDAsset("/tmp/downloaded_usd.usd")
            else:
                usd_asset = USDAsset(fname=scene.assets[asset_name].usd_path)
            usd_asset = usd_asset.scene()
            for geometry_name in usd_asset.get_geometry_names():
                geometry_transform = usd_asset.get_transform(node=geometry_name).tolist()
                geometry_bounds = usd_asset.get_bounds(query=geometry_name).tolist()
                scene_description[asset_name][geometry_name] = dict(
                    geometry_transform=geometry_transform,
                    geometry_bounds=geometry_bounds
                )
        self.assets_description = scene_description
        self.task_description = task_description
        self.action_buffer = []
        self.gripper_action_dict = dict(
            left=torch.as_tensor([1.0,], device='cuda'),
            right=torch.as_tensor([1.0,], device='cuda')
        )
        self.planner = None
        self.action_repeat_counter = 0
        self.next_pose = None

    def grasp_object(self, env, arm, planner):
        frame_transformer_data = env.scene.sensors['object_frame_transformer'].data
        for frame_index, frame_name in enumerate(frame_transformer_data.target_frame_names):
            if arm in frame_name and "grasp" in frame_name:
                target_pose_pos = frame_transformer_data.target_pos_w[..., frame_index, :]
                target_pose_quat = frame_transformer_data.target_quat_w[..., frame_index, :]
                break
        #target_pose_quat = planner.get_ee_pose(joint_state=planner._get_current_joint_state_for_curobo()).quaternion
        print('pos', target_pose_pos)
        print('quat', target_pose_quat)
        target_pose = PoseUtils.make_pose(pos=target_pose_pos, rot=PoseUtils.matrix_from_quat(target_pose_quat))
        #print(target_pose)
        plan_success = planner.update_world_and_plan_motion(target_pose=target_pose, step_size=env.step_dt/2.0, enable_retiming=True)
        input()
        if plan_success == False:
            raise ValueError("Plan failed!")
        while False and planner.has_next_waypoint():
            next_pose = planner.get_next_waypoint_ee_pose().get_matrix()[0]
            if arm == "left":
                target_pose_dict=dict(
                    left=next_pose,
                    right=env.get_robot_eef_pose(eef_name="right")[0]
                )
                self.gripper_action_dict['left'][0] = 1.0
            if arm == "right":
                target_pose_dict=dict(
                    right=next_pose,
                    left=env.get_robot_eef_pose(eef_name="left")[0]
                )
                self.gripper_action_dict['right'][0] = 1.0
            action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_pose_dict,
                gripper_action_dict=self.gripper_action_dict,
            )
            self.action_buffer.append(action.reshape((1, -1)))
        
    def plan_motion_stages(self, env, observation):
        urdf_path = f"{curobo.__path__[0]}/content/assets/robot/openarm_description/openarm_bimanual_v10.urdf"
        tmp_yaml = CuroboPlannerCfg._create_temp_robot_yaml(base_yaml="openarm_bimanual.yml", urdf_path=urdf_path)
        planner_config = CuroboPlannerCfg(
            robot_config_file=tmp_yaml,
            robot_name="openarm",
            debug_planner=True,
            #visualize_plan=True,
            #visualize_spheres=True
        )
        planner = CuroboPlanner(env=env, robot=env.scene['robot'], config=planner_config)
        planner.update_world()
        #planner.update_world()
        print(env.scene.sensors['ee_frame'].data.target_quat_source[..., 0, :])
        #print(planner.get_ee_pose(joint_state=planner._get_current_joint_state_for_curobo()).quaternion)
        #exit()
        self.grasp_object(
            env=env,
            arm='left',
            planner=planner
        )
        self.arm = "left"
        self.planner = planner
        self.plan_index = 0

    def plan_motion(self, env, observation):
        state = env.scene.get_state(is_relative=True)
        sim_state_description = {}
        for asset_type in state.keys():
            sim_state_description[asset_type] = {}
            for state_tensor_key in state[asset_type].keys():
                sim_state_description[asset_type][state_tensor_key] = {}
                for key2 in state[asset_type][state_tensor_key].keys():
                    sim_state_description[asset_type][state_tensor_key][key2] = state[asset_type][state_tensor_key][key2].flatten(start_dim=1).detach().cpu().numpy().tolist()
        sim_state_description['object_frames'] = {}
        frame_transformer_data = env.scene.sensors['object_frame_transformer'].data
        for frame_index, frame_name in enumerate(frame_transformer_data.target_frame_names):
            sim_state_description['object_frames'][frame_name] = dict(
                target_pos_world=frame_transformer_data.target_pos_w[..., frame_index, :].detach().cpu().numpy().flatten().tolist(),
                target_quat_world=frame_transformer_data.target_quat_w[..., frame_index, :].detach().cpu().numpy().flatten().tolist(),
            )
        prompt = """Given the geometry of the following USD assets (the transforms and bounds are relative to the origin of each asset): {assets_description}
And given the following IsaacLab simulation state: {sim_state_description}
Predict the next waypoint for one of the robotic arms (left or right) to follow in order to execute the following task: {task_description}
Answer in the following format:

{{"arm_side": "...(left or right)", "gripper_state": "...(open or close)", "tcp_pose_xyz_wxyz": [..., ..., ..., ..., ..., ..., ...]}}"""
        response = litellm.completion(
            model='ollama/qwen3',
            messages=[
                {"role": "user", "content": prompt.format(
                    assets_description=self.assets_description,
                    sim_state_description=sim_state_description,
                    task_description=self.task_description
                )}
            ],
            reasoning_effort='none'
        ).choices[0].message.content
        print(response)
        #exit()
        response = response.strip('```json').strip('```').strip('\n')
        next_waypoint = json.loads(response)
        urdf_path = f"{curobo.__path__[0]}/content/assets/robot/openarm_description/openarm_bimanual_v10.urdf"
        if "left" in next_waypoint['arm_side']:
            tmp_yaml = CuroboPlannerCfg._create_temp_robot_yaml(base_yaml="openarm_bimanual.yml", urdf_path=urdf_path)
        if "right" in next_waypoint['arm_side']:
            tmp_yaml = CuroboPlannerCfg._create_temp_robot_yaml(base_yaml="openarm_bimanual.yml", urdf_path=urdf_path)
        planner_config = CuroboPlannerCfg(
            robot_config_file=tmp_yaml,
            robot_name="openarm",
            debug_planner=True
        )
        if "left" in next_waypoint['arm_side']:
            planner_config.ee_link_name = "openarm_left_ee_tcp"
            #planner_config.gripper_joint_names=["openarm_left_finger_joint1", "openarm_left_finger_joint2"],
            #planner_config.gripper_open_positions={"openarm_left_finger_joint1": 0.04, "openarm_left_finger_joint2": 0.04},
            #planner_config.gripper_closed_positions={"openarm_left_finger_joint1": 0.023, "openarm_left_finger_joint2": 0.023},
            #planner_config.hand_link_names=["openarm_left_left_finger", "openarm_left_right_finger", "openarm_left_hand"],
        if "right" in next_waypoint['arm_side']:
            planner_config.ee_link_name = "openarm_right_ee_tcp"
        planner = CuroboPlanner(env=env, robot=env.scene['robot'], config=planner_config)
        target_pose_pos = torch.tensor(next_waypoint["tcp_pose_xyz_wxyz"][:3], dtype=torch.float32)
        target_pose_quat = torch.tensor(next_waypoint["tcp_pose_xyz_wxyz"][3:7], dtype=torch.float32)
        target_pose = PoseUtils.make_pose(pos=target_pose_pos, rot=PoseUtils.matrix_from_quat(target_pose_quat))
        #print(target_pose)
        plan_success = planner.update_world_and_plan_motion(target_pose=target_pose)
        if plan_success == False:
            raise ValueError("Plan failed!")
        while planner.has_next_waypoint():
            next_pose = planner.get_next_waypoint_ee_pose()
            if "left" in next_waypoint['arm_side']:
                target_pose_dict=dict(
                    left=next_pose,
                    right=env.get_robot_eef_pose(eef_name="right")[0]
                )
                if "close" in next_waypoint['gripper_state']:
                    self.gripper_action_dict['left'][0] = -1.0
                if "open" in next_waypoint['gripper_state']:
                    self.gripper_action_dict['left'][0] = 1.0
            if "right" in next_waypoint['arm_side']:
                target_pose_dict=dict(
                    right=next_pose,
                    left=env.get_robot_eef_pose(eef_name="left")[0]
                )
                if "close" in next_waypoint['gripper_state']:
                    self.gripper_action_dict['right'][0] = -1.0
                if "open" in next_waypoint['gripper_state']:
                    self.gripper_action_dict['right'][0] = 1.0
            action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_pose_dict,
                gripper_action_dict=self.gripper_action_dict,
            )
            self.action_buffer.append(action)

    def get_action(self, env, observation):
        if self.planner == None or self.planner.current_plan == None or len(self.planner.current_plan) <= self.plan_index:
            self.plan_motion_stages(env=env, observation=observation)
        '''if self.action_repeat_counter >= 1 or self.next_pose == None:
            self.next_pose = self.planner.get_next_waypoint_ee_pose().get_matrix()[0]
            self.action_repeat_counter = 0'''
        target_joint_pos = self.planner.current_plan[self.plan_index].position
        action = torch.zeros((1, 16), device=env.device)
        action[0, 0:7] = target_joint_pos[0:7]
        action[0, 7] = 1.0
        #action[0, 8:15] = target_joint_pos[9:16]
        action[0, 15] = 1.0
        self.plan_index += 1
        return action
        if self.arm == "left":
                target_pose_dict=dict(
                    left=self.next_pose,
                    right=env.get_robot_eef_pose(eef_name="right")[0]
                )
                self.gripper_action_dict['left'][0] = 1.0
        if self.arm == "right":
                target_pose_dict=dict(
                    right=self.next_pose,
                    left=env.get_robot_eef_pose(eef_name="left")[0]
                )
                self.gripper_action_dict['right'][0] = 1.0
        action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_pose_dict,
                gripper_action_dict=self.gripper_action_dict,
        )
        self.action_repeat_counter += 1
        return action.reshape((1, -1))
    
    def reset(self, env_ids = None):
        self.planner = None