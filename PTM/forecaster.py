from utils import linear_fitting_solver, Fourier_fitting_solver, sin_fitting_solver, cos_fitting_solver
from symfit import parameters, variables, sin, cos, Fit
import numpy as np
import pandas as pd
import pmdarima as pm

class forecaster_arima():
    def __init__(self, NS_variable_dim, ahead_length=1):
        self.ns_var_length = NS_variable_dim
        self.predict_length = ahead_length
        self.data_dic , self.model_dic  = self._set_forecaster_dic()

    
    def _set_forecaster_dic(self) :
        data_dic, model_dic = {}, {}
        for i in range(self.ns_var_length) : 
            data_dic["dim_"+str(i+1)] = []
            model_dic["dim_"+str(i+1)] = None
        return data_dic , model_dic

    def update_nonstationary_variable(self,ep,noisy_NS_var : list):
        del ep
        assert len(noisy_NS_var) == self.ns_var_length
        for i in range(self.ns_var_length) :
            self.data_dic["dim_"+str(i+1)].append(noisy_NS_var[i])

    def fit_forecastor(self):
        for i in range(self.ns_var_length) : 
            df = pd.Series(self.data_dic["dim_"+str(i+1)])
            model = pm.auto_arima(df, start_p=1, start_q=1,
                                  test='adf',  # use adftest to find optimal 'd'
                                  max_p=10, max_q=10,  # maximum p and q
                                  m=1,  # frequency of series
                                  d=None,  # let model determine 'd'
                                  seasonal=False,  # No Seasonality
                                  start_P=0,
                                  D=0,
                                  trace=True,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)
            self.model_dic["dim_"+str(i+1)] = model

    def predict_nonstationary_variable(self,current_episode):
        del current_episode
        # Forecast
        n_periods = self.predict_length
        assert n_periods == 1
        future_ns_var = []
        for i in range(self.ns_var_length) :
            fc, confint = self.model_dic["dim_"+str(i+1)].predict(n_periods=n_periods, return_conf_int=True)
            index_of_fc = np.arange(len(self.data_dic["dim_" + str(i + 1)]), len(self.data_dic["dim_" + str(i + 1)]) + n_periods)
            # make series for plotting purpose
            # fc_series = pd.Series(fc, index=index_of_fc)
            lower_series = pd.Series(confint[:, 0], index=index_of_fc)
            upper_series = pd.Series(confint[:, 1], index=index_of_fc)

            future_ns_var.append(fc.values[0])

        return future_ns_var


    def save_model(self,path):
        import pickle
        for i in range(self.ns_var_length) :
            with open(path+'/arima_nsVARdim_'+str(i+1)+".pkl", 'wb') as f:
                pickle.dump(self.model_dic["dim_"+str(i+1)], f)

                
class forecaster_arima_manual():
    def __init__(self, NS_variable_dim,p,d,q, ahead_length=1):
        self.ns_var_length = NS_variable_dim
        self.predict_length = ahead_length
        self.data_dic , self.model_dic  = self._set_forecaster_dic()
        self.p , self.d , self.q = p,d,q

    
    def _set_forecaster_dic(self) :
        data_dic, model_dic = {}, {}
        for i in range(self.ns_var_length) : 
            data_dic["dim_"+str(i+1)] = []
            model_dic["dim_"+str(i+1)] = None
        return data_dic , model_dic

    def update_nonstationary_variable(self,ep,noisy_NS_var : list):
        del ep
        assert len(noisy_NS_var) == self.ns_var_length
        for i in range(self.ns_var_length) :
            self.data_dic["dim_"+str(i+1)].append(noisy_NS_var[i])

    def fit_forecastor(self):
        for i in range(self.ns_var_length) : 
            df = pd.Series(self.data_dic["dim_"+str(i+1)])
            model = pm.ARIMA(order=(self.p, self.d, self.q))
            self.model_dic["dim_"+str(i+1)] = model

    def predict_nonstationary_variable(self,current_episode):
        del current_episode
        # Forecast
        n_periods = self.predict_length
        assert n_periods == 1
        future_ns_var = []
        for i in range(self.ns_var_length) :
            fc, confint = self.model_dic["dim_"+str(i+1)].predict(n_periods=n_periods, return_conf_int=True)
            index_of_fc = np.arange(len(self.data_dic["dim_" + str(i + 1)]), len(self.data_dic["dim_" + str(i + 1)]) + n_periods)
            # make series for plotting purpose
            # fc_series = pd.Series(fc, index=index_of_fc)
            lower_series = pd.Series(confint[:, 0], index=index_of_fc)
            upper_series = pd.Series(confint[:, 1], index=index_of_fc)

            future_ns_var.append(fc.values[0])

        return future_ns_var


    def save_model(self,path):
        import pickle
        for i in range(self.ns_var_length) :
            with open(path+'/arima_nsVARdim_'+str(i+1)+".pkl", 'wb') as f:
                pickle.dump(self.model_dic["dim_"+str(i+1)], f)
                
class forecaster_simpleaverage():
    def __init__(self, NS_variable_dim, ahead_length=1,sliding_windew_length=10):
        self.ns_var_length = NS_variable_dim
        self.predict_length = ahead_length
        self.sliding_windew_length =  sliding_windew_length
        self.data_dic , self.model_dic  = self._set_forecaster_dic()

    def _set_forecaster_dic(self) :
        data_dic, model_dic = {}, {}
        for i in range(self.ns_var_length) : 
            data_dic["dim_"+str(i+1)] = []
            model_dic["dim_"+str(i+1)] = None
        return data_dic , model_dic

    def update_nonstationary_variable(self,ep,noisy_NS_var : list):
        del ep
        assert len(noisy_NS_var) == self.ns_var_length
        for i in range(self.ns_var_length) :
            self.data_dic["dim_"+str(i+1)].append(noisy_NS_var[i])

    def fit_forecastor(self):
        pass

    def predict_nonstationary_variable(self,current_episode):
        del current_episode
        # Forecast
        n_periods = self.predict_length
        assert n_periods == 1
        future_ns_var = []
        for i in range(self.ns_var_length) :
            current_length = len(self.data_dic["dim_"+str(i+1)])
            min_length = max(1,current_length-self.sliding_windew_length)
            future_ns_var.append(np.mean(self.data_dic["dim_"+str(i+1)][min_length-1:]))

        return future_ns_var


    def save_model(self,path):
        import pickle
        for i in range(self.ns_var_length) :
            with open(path+'/arima_nsVARdim_'+str(i+1)+".pkl", 'wb') as f:
                pickle.dump(self.model_dic["dim_"+str(i+1)], f)


class forecaster():
    def __init__(self,NS_variable_dim,ahead_length=1,solvertype="fourier",order=1):
        self.num_of_forecastor = NS_variable_dim
        # self.function_fitting_length = len(self.list_of_ep)
        self.predict_length = ahead_length
        self.dic = self._set_forecator_dic()
        self.list_of_ep = []
        if solvertype == "linear" :
            self.fitting_solver = linear_fitting_solver()
        elif solvertype == "fourier" :
            self.fitting_solver = Fourier_fitting_solver(order)
        elif solvertype == "sin" :
            self.fitting_solver = sin_fitting_solver(order)
        else :
            raise Exception("solver type error")

    def _set_forecator_dic(self):
        dic = {}
        for i in range(self.num_of_forecastor) :
            dic["forcastor_"+str(i)] = None
            dic["forcastor_"+str(i)+"_inputY"] = []
            dic["forcastor_"+str(i)+"_result"] = None

        return dic

    def update_nonstationary_variable(self,ep,noisy_NS_var : list):
        self.list_of_ep.append(ep)
        assert len(noisy_NS_var) == self.num_of_forecastor
        for i in range(self.num_of_forecastor) :
            self.dic["forcastor_" + str(i) + "_inputY"].append(noisy_NS_var[i])

    def fit_forecastor(self):
        self.function_fitting_length = len(self.list_of_ep)
        for i in range(self.num_of_forecastor) :
            for _ in range(300) :
                self.dic["forcastor_"+str(i)] = \
                    Fit(self.fitting_solver.model_dict, x = np.array([x for x in self.list_of_ep[-self.function_fitting_length:]]),\
                        y=np.array(self.dic["forcastor_" + str(i) + "_inputY"][-self.function_fitting_length:]))
                self.dic["forcastor_" + str(i) + "_result"] = self.dic["forcastor_"+str(i)].execute()
                if self.dic["forcastor_" + str(i) + "_result"].gof_qualifiers['r_squared'] > 0.9 :
                    self.reset_init_forecaster()
                    break
                else :
                    print("redo forecastor..")
                    self.random_reset_init_forecaster()
        # print(self.dic["forcastor_" + str(i) + "_result"].gof_qualifiers)

        # print(self.dic["forcastor_" + str(i) + "_result"].params)


    def reset_init_forecaster(self):
        for i in range(self.num_of_forecastor) :
            self.fitting_solver.w.value = self.dic["forcastor_" + str(i) + "_result"].params['w']
            # print(self.dic["forcastor_" + str(i) + "_result"].params['w'])
            for k in self.fitting_solver.coeff.keys() :
                self.fitting_solver.coeff[k].value = self.dic["forcastor_" + str(i) + "_result"].params[k]
                # print(self.dic["forcastor_" + str(i) + "_result"].params[k])

    def random_reset_init_forecaster(self):
        for i in range(self.num_of_forecastor) :
            self.fitting_solver.w.value = np.random.uniform(-1,1)
            # self.fitting_solver.b0.value = np.random.uniform(0,1)

    def predict_nonstationary_variable(self,current_episode):
        future_ns_var = []
        for i in range(self.num_of_forecastor) :
            ns_var = self.dic["forcastor_" + str(i)].model(x = np.array([current_episode+self.predict_length]),
                                                  **self.dic["forcastor_" + str(i) + "_result"].params).y
            future_ns_var.append(ns_var[0])

        return future_ns_var

    def save_model(self,path):
        import json
        for i in range(self.num_of_forecastor) :
            json_file = json.dumps(self.dic["forcastor_" + str(i) + "_result"].params)
            f = open(path +"/f_"+str(i)+".json", "w")
            f.write(json_file)
            f.close()
