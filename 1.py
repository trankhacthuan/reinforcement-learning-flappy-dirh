class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = self.state

        # 0 = Up, 1 = Down, 2 = Left, 3 = Right
        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1

        # Không cho đi ra ngoài biên
        row = max(0, min(row, self.rows - 1))
        col = max(0, min(col, self.cols - 1))

        next_state = (row, col)

        # Cliff: hàng cuối, từ cột 1 đến 10
        if row == 3 and 1 <= col <= 10:
            reward = -100
            next_state = self.start
            done = False
        elif next_state == self.goal:
            reward = -1
            done = True
        else:
            reward = -1
            done = False

        self.state = next_state
        return next_state, reward, done