from maze_environment import MazeGrid
from q_learning_table import QLambdaLearning
import matplotlib.pyplot as plt


def update():
    learning_info = []
    for episode in range(100):
        state = env.reset()
        total_rewards = 0
        while True:
            env.render()

            # choose an action from what we learned so far from q table
            action = q_rl.choose_action(str(state))
            next_state, reward, done = env.step(action)
            total_rewards += reward
            # update q table and learn based on movement
            q_rl.q_lambda(str(state), action, reward, str(next_state))
            state = next_state

            if done:
                learning_info.append((episode, total_rewards))
                break

    # end of the game and learning and save the result
    q_rl.Q.to_csv("./learning_results/q_result.csv")
    print('Game is Finished After 100 Episode.')
    # close the environment
    env.destroy()
    plt.plot(*zip(*learning_info))
    plt.ylabel('rewards')
    plt.xlabel('episodes')
    plt.show()


if __name__ == "__main__":
    env = MazeGrid()
    action_indexes = list(range(env.n_actions))
    q_rl = QLambdaLearning(actions=action_indexes)

    env.after(100, update)
    env.mainloop()




