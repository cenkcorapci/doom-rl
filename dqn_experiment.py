import itertools
from random import randint, random
from time import time, sleep
import sys

from tqdm import trange
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

from models.dqn import *

FLAGS = flags.FLAGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default configuration file path
default_config_file_path = "vizdoom/scenarios/basic.cfg"


def find_eps(epoch):
    """Balance exploration and exploitation as we keep learning"""
    start, end = 1.0, 0.1
    const_epochs, decay_epochs = .1 * FLAGS.epochs, .6 * FLAGS.epochs
    if epoch < const_epochs:
        return start
    elif epoch > decay_epochs:
        return end
    # Linear decay
    progress = (epoch - const_epochs) / (decay_epochs - const_epochs)
    return start - progress * (start - end)


def perform_learning_step(epoch, game, model, actions):
    s1 = game_state(game)
    if random() <= find_eps(epoch):
        a = torch.tensor(randint(0, len(actions) - 1)).long()
    else:
        s1 = s1.reshape([1, 1, *resolution])
        a = model.get_best_action(s1.to(device))
    reward = game.make_action(actions[a], frame_repeat)

    if game.is_episode_finished():
        isterminal, s2 = 1., None
    else:
        isterminal = 0.
        s2 = game_state(game)

    model.memory.add_transition(s1, a, s2, isterminal, reward)
    model.learn_from_memory()


def initialize_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    return game


def evaluate(iters, game, model, actions):
    scores = np.array([])
    for _ in trange(FLAGS.test_episodes, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = game_state(game)
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            a_idx = model.get_best_action(state.to(device))
            game.make_action(actions[a_idx], frame_repeat)
        r = game.get_total_reward()
        scores = np.append(scores, r)
    print(f'Results: mean: {scores.mean():.1f} +/- {scores.std():.1f}')


def train(game, model, actions):
    time_start = time()
    print("Saving the network weigths to:", FLAGS.save_path)
    for epoch in range(FLAGS.epochs):
        print(f'Epoch {epoch + 1}')
        episodes_finished = 0
        scores = np.array([])
        game.new_episode()
        for learning_step in trange(FLAGS.iters, leave=False):
            perform_learning_step(epoch, game, model, actions)
            if game.is_episode_finished():
                score = game.get_total_reward()
                scores = np.append(scores, score)
                game.new_episode()
                episodes_finished += 1
        print(f'Completed {episodes_finished} episodes')
        print(f'Mean: {scores.mean():.1f} +/- {scores.std():.1f}')
        print("Testing...")
        evaluate(FLAGS.test_episodes, game, model, actions)
        torch.save(model, FLAGS.save_path)
    print(f'Total elapsed time: {(time() - time_start):.2f} minutes')


def watch_episodes(game, model, actions):
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    for episode in range(FLAGS.watch_episodes):
        game.new_episode(f'episode-{episode}')
        while not game.is_episode_finished():
            state = game_state(game)
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            a_idx = model.get_best_action(state.to(device))
            game.set_action(actions[a_idx])
            for _ in range(frame_repeat):
                game.advance_action()
        sleep(1.0)
        score = game.get_total_reward()
        print(f'Total score: {score}')


if __name__ == '__main__':
    flags.DEFINE_integer('batch_size', 64, 'Batch size')
    flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate')
    flags.DEFINE_float('discount', 0.99, 'Discount factor')
    flags.DEFINE_integer('replay_memory', 10000, 'Replay memory capacity')
    flags.DEFINE_integer('epochs', 20, 'Number of epochs')
    flags.DEFINE_integer('iters', 2000, 'Iterations per epoch')
    flags.DEFINE_integer('watch_episodes', 10, 'Trained episodes to watch')
    flags.DEFINE_integer('test_episodes', 100, 'Episodes to test with')
    flags.DEFINE_string('config', default_config_file_path,
                        'Path to the config file')
    flags.DEFINE_boolean('skip_training', False, 'Set to skip training')
    flags.DEFINE_boolean('load_model', False, 'Load the model from disk')
    flags.DEFINE_string('save_path', 'model-doom.pth',
                        'Path to save/load the model')
    FLAGS(sys.argv)
    game = initialize_vizdoom(FLAGS.config)

    n = game.get_available_buttons_size()
    actions = [list(a) for a in itertools.product([0, 1], repeat=n)]

    if FLAGS.load_model:
        print(f'Loading model from: {FLAGS.save_path}')
        model = torch.load(FLAGS.save_path).to(device)
    else:
        model = QNet(len(actions)).to(device)

    print("Starting the training!")
    if not FLAGS.skip_training:
        train(game, model, actions)

    game.close()
    print("======================================")
    watch_episodes(game, model, actions)
