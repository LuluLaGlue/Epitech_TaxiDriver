import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog

from SARSA.SARSA_Train import train as train_S
from ValueIteration.VI_Train import train as train_VI
from QLearning.QLearning_Train import train as train_QL
from MonteCarlo.MC_Train import monte_carlo_e_soft as train_MC


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.args = {}
        mode = self.__getMode()
        algo = self.__getAlgo()
        self.setParam(algo)
        if mode == "Custom":
            self.askParam()
        print("Using {} mode with {} algorithm.".format(mode, algo))
        print("Selected Parameters: ", self.args)
        self.train(algo)

    def __getMode(self) -> str:
        '''
        Display a dropdown selector with 2 modes (custom and performance)
        
        Returns the selected mode
        '''
        return self.__getChoice(("Custom", "Performance"))

    def __getAlgo(self) -> str:
        '''
        Display a dropdown selector with a list of available algorithms
        
        Returns the name of the selected algorithm
        '''
        return self.__getChoice(
            ("Value Iteration", "Monte Carlo", "Q-Learning", "SARSA"))

    def setParam(self, algo: str):
        '''
        Based on the given algorithm name, creates a list of paramaters and display an input field for each  param.
        '''
        if algo == "Q-Learning" or algo == "SARSA":
            self.args = {
                "lr": 0.01 if algo == "Q-Learning" else 0.85,
                "episodes": 25000 if algo == "Q-Learning" else 10000,
                "gamma": 0.99,
                "epsilon": 1,
                "min_epsilon": 0.001,
                "decay_rate": 0.01
            }
        elif algo == "Monte Carlo":
            self.args = {"episodes": 500000, "epsilon": 0.01}
        elif algo == "Value Iteration":
            self.args = {"gamma": 0.9, "significant_improvement": 0.001}

    def askParam(self):
        for key in self.args:
            if key == "episodes":
                self.__getInteger("Number of Episodes",
                                  key,
                                  default=self.args[key],
                                  min=1,
                                  max=1000000)
            else:
                self.__getDouble(key,
                                 key,
                                 default=self.args[key],
                                 min=0,
                                 max=1,
                                 decimals=4)

    def train(self, algo: str):
        '''
        Based on the selected algorithm, trains it with the previously selected parameters.
        '''
        plt.ion()
        if algo == "Value Iteration":
            train_VI(
                gamma=self.args["gamma"],
                significant_improvement=self.args["significant_improvement"],
                path="ValueIteration/v-iteration")
        elif algo == "Monte Carlo":
            train_MC(episodes=self.args["episodes"],
                     epsilon=self.args["epsilon"],
                     path="MonteCarlo/policy.pkl")
        elif algo == "Q-Learning":
            train_QL(episodes=self.args["episodes"],
                     lr=self.args["lr"],
                     gamma=self.args["gamma"],
                     epsilon=self.args["epsilon"],
                     max_epsilon=self.args["epsilon"],
                     min_epsilon=self.args["min_epsilon"],
                     epsilon_decay=self.args["decay_rate"],
                     show_empty=False,
                     path_table="QLearning/qtable",
                     path_graph="QLearning/QLearning_graph.png")
        elif algo == "SARSA":
            train_S(episodes=self.args["episodes"],
                    gamma=self.args["gamma"],
                    epsilon=self.args["epsilon"],
                    max_epsilon=self.args["epsilon"],
                    min_epsilon=self.args["min_epsilon"],
                    epsilon_decay=self.args["decay_rate"],
                    alpha=self.args["lr"],
                    path_table="SARSA/qtable",
                    path_graph="SARSA/SARSA_graph.png")
        plt.ioff()

    def __getChoice(self, items: tuple[str]) -> str:
        item, okPressed = QInputDialog.getItem(self, "Get item", "Algorithm:",
                                               items, 0, False)
        if okPressed and item:
            return item

    def __getDouble(self, placeholder: str, target: str, default: float,
                    min: float, max: float, decimals: int):
        d, okPressed = QInputDialog.getDouble(self, placeholder, placeholder,
                                              default, min, max, decimals)
        if okPressed:
            self.args[target] = d

    def __getInteger(self, placeholder: str, target: str, default: int,
                     min: int, max: int):
        d, okPressed = QInputDialog.getInt(self, placeholder, "Value:",
                                           default, min, max, 100)
        if okPressed:
            self.args[target] = d


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
