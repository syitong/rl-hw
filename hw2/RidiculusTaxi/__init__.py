from gym.envs.registration import register

register(
    id='Taxi-v3',
    entry_point='RidiculusTaxi.ridiculustaxi:TaxiEnv',)
