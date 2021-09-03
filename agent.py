import numpy as np
import utils

class Agent:
    #에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2 #주식 보유 비율, 포트폴리오 가치 비율

    #매매 수수료 및 세금
    TRADING_CHARGE = 0.00015 #거래 수수료 (일반적으로 0.015%)
    TRADING_TAX = 0.0025 # 거래세 (실제 0.25%)
    #TRADING_CHARGE = 0 # 거래 수수료 미적용일때
    #TRADING_TAX = 0 # 거래세 미적용

    #행동
    ACTION_BUY = 0 # 매수
    ACTION_SELL = 1 # 매도
    ACTION_HOLD = 2 # 홀딩
    #인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS) # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self , environment, min_trading_unit = 1 , max_trading_unit = 2, delayed_reward_threshold = .05):
        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment

        #최소 매매 단위, 최대 매매 단위
        self.min_trading_unit = min_trading_unit # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit # 최대 단일 거래 단위

        #지연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        #Agent 클래스의 속성
        self.initial_balance = 0 # 초기 자본금
        self.balance = 0 # 현재 현금 잔고
        self.num_stocks = 0 # 보유 주식 수
        # PV = balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.base_portfolio_value = 0 # 직전 학습 시점의 PV
        self.num_buy = 0 # 매수 횟수
        self.num_sell = 0 # 매도 횟수
        self.num_hold = 0 # 홀딩 횟수
        self.immediate_reward = 0 #즉시 보상
        self.profitloss = 0 # 현재 손익
        self.base_profitloss = 0 # 직전 지연 보상 이후 손익
        self.exploration_base = 0 # 탐험 행동 결정 기준

        # Agent 클래스의 상태
        self.ratio_hold = 0 # 주식 보유 비율
        self.ratio_portfolio_value = 0 # 포트폴리오 가치 비율

        def reset(self):
            self.balance = self.initial_balance
            self.num_stocks = 0
            self.portfolio_value = self.inital_balance
            self.base_portfolio_value = self.initial_balance
            self.num_buy = 0
            self.num_sell = 0
            self.num_hold = 0
            self.immediate_reward = 0
            self.ratio_hold = 0
            self.ratio_portfolio_value = 0

        def reset_exploration(self):
            self.exploration_base = 0.5 + np.random.rand() / 2

        def set_balance(self, balance):
            self.initial_balance = balance

        def get_states(self):
            self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())
            self.ratio_portfolio_value = (self.portfolio_value / self.base_portfolio_value)
            return(
                self.ratio_hold,
                self.ratio_portfolio_value
            )

        def decide_action(self, pred_value, pred_policy, epsilon):
            confidence = 0.

            pred = pred_policy
            if pred is None:
                pred = pred_value

            if pred is None:
                #예측 값이 없을 경우 탐험
                epsilon = 1
            else:
                # 값이 모두 같은 경우 탐험
                maxpred = np.max(pred) #최댓값 뽑아서
                if(pred == maxpred).all(): #행렬과 비교(스칼라가 행렬로 넓혀짐)한 값이 모두 참이면
                    epsilon = 1

            # 탐험 결정
            if np.random.rand() < epsilon:
                exploration = True
                if np.random.rand() < self.exploration_base:
                    action = self.ACTION_BUY #매수
                else:
                    action = np.random.randint(self.NUM_ACTIONS - 1) + 1 #매도 혹은 홀드
            else:
                exploration = False
                action = np.argmax(pred)

            confidence = .5
            if pred_policy is not None:
                confidence = pred[action]
            elif pred_value is not None:
                confidence = utils.sigmoid(pred[action])

            return action , confidence , exploration

        def validate_action(self, action):
            if action == Agent.ACTION_BUY:
                #적어도 1주를 살 수 있는지 확인
                if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                    return False
            elif action == Agent.ACTION_SELL:
                #주식 잔고가 있는지 확인
                if self.num_stocks <= 0:
                    return False
            return True

        




