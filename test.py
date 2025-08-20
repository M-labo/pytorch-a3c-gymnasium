import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic
from gymnasium.wrappers import RecordVideo

def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    every = max(1, int(args.video_every))
    if args.rec_video == True:
        name_prefix = f"pong_a3c_gymnasium_{int(counter.value):03d}"
        env = RecordVideo(
            env,
            args.video_dir,
            # episode_trigger=lambda ep_no: True,   # 1 エピソード目だけ録画
            episode_trigger=lambda ep: ep % (args.video_every) == 0,
            # episode_trigger=lambda ep_no: ep_no == 0,   # 1 エピソード目だけ録画
            name_prefix=name_prefix
        )
    # env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state, info = env.reset(seed=args.seed + rank)
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, terminated, truncated, _ = env.step(action[0, 0])
        done = terminated or truncated or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state, info = env.reset(seed=args.seed+rank)
            time.sleep(60)

        state = torch.from_numpy(state)
