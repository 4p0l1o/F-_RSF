import utils
from plato.config import Config
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import seaborn as sns
import statistics
from scipy import stats


def main():
    trees = [10, 100, 1000]
    clients = [2, 5, 10]

    client_Ibs = []
    client_Ci = []
    FedIbsC_Ibs =[]
    FedIbsC_Ci =[]
    FedCiC_Ibs = []
    FedCiC_Ci = []
    FedIbsS_Ibs =[]
    FedIbsS_Ci =[]
    FedCiS_Ibs = []
    FedCiS_Ci = []
    client_Ibs_std = []
    client_Ci_std = []
    FedIbsC_Ibs_std =[]
    FedIbsC_Ci_std =[]
    FedCiC_Ibs_std = []
    FedCiC_Ci_std = []
    FedIbsS_Ibs_std =[]
    FedIbsS_Ci_std =[]
    FedCiS_Ibs_std = []
    FedCiS_Ci_std = []
    baseline_ci_list = []
    baseline_ibs_list =[]
    baseLine_path = (
                        f"{Config().params['result_path']}/{Config.data.datasource}/Baseline_accuracy.csv"
                    )
    baseline = utils.read_csv(baseLine_path)

    for row in baseline:
        rowlist = row[0].split(",")
        if rowlist[0] == "baseline":
            baseline_ci_list.append(float(rowlist[1]))
            baseline_ibs_list.append(float(rowlist[2]))    
    print("baseline stds:")
    print(round(statistics.stdev(baseline_ci_list)*100,2))
    print(round(statistics.stdev(baseline_ibs_list)*100,2))
    baseline_ci = round(100*sum(baseline_ci_list)/len(baseline_ci_list),2)
    baseline_ibs = round(100*sum(baseline_ibs_list)/len(baseline_ibs_list),2)
    print("baseline:")
    print(baseline_ci)
    print(baseline_ibs)

    for client in clients:
        tempclient_Ibs = []
        tempclient_Ci = []
        tempFedIbsC_Ibs =[]
        tempFedIbsC_Ci =[]
        tempFedCiC_Ibs = []
        tempFedCiC_Ci = []
        tempFedIbsS_Ibs =[]
        tempFedIbsS_Ci =[]
        tempFedCiS_Ibs = []
        tempFedCiS_Ci = []
        tempclient_Ibs_std = []
        tempclient_Ci_std = []
        tempFedIbsC_Ibs_std =[]
        tempFedIbsC_Ci_std =[]
        tempFedCiC_Ibs_std = []
        tempFedCiC_Ci_std = []
        tempFedIbsS_Ibs_std =[]
        tempFedIbsS_Ci_std =[]
        tempFedCiS_Ibs_std = []
        tempFedCiS_Ci_std = []
        for tree in trees:
            path = (
                f"{Config().params['result_path']}/{Config.data.datasource}/accuracy_{client}_clients_{int(tree/client)}_trees.csv"
            )
            print(path)
            if not os.path.isfile(path):
                print("no path")
                tempclient_Ibs.append(0)
                tempclient_Ci.append(0)
                tempFedIbsC_Ibs.append(0)
                tempFedIbsC_Ci.append(0)
                tempFedCiC_Ibs.append(0)
                tempFedCiC_Ci.append(0)
                tempFedIbsS_Ibs.append(0)
                tempFedIbsS_Ci.append(0)
                tempFedCiS_Ibs.append(0)
                tempFedCiS_Ci.append(0)
                tempclient_Ibs_std.append(0)
                tempclient_Ci_std.append(0)
                tempFedIbsC_Ibs_std.append(0)
                tempFedIbsC_Ci_std.append(0)
                tempFedCiC_Ibs_std.append(0)
                tempFedCiC_Ci_std.append(0)
                tempFedIbsS_Ibs_std.append(0)
                tempFedIbsS_Ci_std.append(0)
                tempFedCiS_Ibs_std.append(0)
                tempFedCiS_Ci_std.append(0)
                continue
            
            else:
                ttempclient_Ibs = []
                ttempclient_Ci = []
                ttempFedIbsC_Ibs =[]
                ttempFedIbsC_Ci =[]
                ttempFedCiC_Ibs = []
                ttempFedCiC_Ci = []
                ttempFedIbsS_Ibs =[]
                ttempFedIbsS_Ci =[]
                ttempFedCiS_Ibs = []
                ttempFedCiS_Ci = []
                models = ttempclient_Ibs, ttempclient_Ci, ttempFedIbsC_Ibs, ttempFedIbsC_Ci, ttempFedCiC_Ibs, ttempFedCiC_Ci, ttempFedIbsS_Ibs, ttempFedIbsS_Ci, ttempFedCiS_Ibs, ttempFedCiS_Ci
                params = utils.read_csv(path)
                for row in params:
                    rowlist = row[0].split(",")
                    #print(rowlist)
                    if rowlist[0] == "":
                        continue
                    if rowlist[0].startswith("Fed_IBS_client"):
                        ttempFedIbsC_Ibs.append(float(rowlist[2]))
                        ttempFedIbsC_Ci.append(float(rowlist[1]))
                    elif rowlist[0].startswith("FED_CI_client"):
                        ttempFedCiC_Ibs.append(float(rowlist[2]))
                        ttempFedCiC_Ci.append(float(rowlist[1]))
                    elif rowlist[0].startswith("Federated_ibs_S"):
                        ttempFedIbsS_Ibs.append(float(rowlist[2]))
                        ttempFedIbsS_Ci.append(float(rowlist[1]))
                    elif rowlist[0].startswith("Federated_ci_S"):
                        ttempFedCiS_Ibs.append(float(rowlist[2]))
                        ttempFedCiS_Ci.append(float(rowlist[1]))
                    elif rowlist[0].startswith("client"):
                        continue
                    else:
                        if not len(rowlist)>2:
                            print(client)
                            print(tree)
                        ttempclient_Ibs.append(float(rowlist[2]))
                        ttempclient_Ci.append(float(rowlist[1]))
                
                print("performing t-test for CI, client, tree: ", client, tree)
                make_t_test(ttempclient_Ci, baseline_ci_list, ttempFedIbsS_Ci, ttempFedCiS_Ci)
                print("performing t-test for IBS, client, tree: ", client, tree)
                make_t_test(ttempclient_Ibs, baseline_ibs_list, ttempFedIbsS_Ibs, ttempFedCiS_Ibs)

                tempclient_Ibs.append(round(100*sum(ttempclient_Ibs)/len(ttempclient_Ibs),2))
                tempclient_Ci.append(round(100*sum(ttempclient_Ci)/len(ttempclient_Ci),2))
                tempFedIbsC_Ibs.append(round(100*sum(ttempFedIbsC_Ibs)/len(ttempFedIbsC_Ibs),2))
                tempFedIbsC_Ci.append(round(100*sum(ttempFedIbsC_Ci)/len(ttempFedIbsC_Ci),2))
                tempFedCiC_Ibs.append(round(100*sum(ttempFedCiC_Ibs)/len(ttempFedCiC_Ibs),2))
                tempFedCiC_Ci.append(round(100*sum(ttempFedCiC_Ci)/len(ttempFedCiC_Ci),2))
                tempFedIbsS_Ibs.append(round(100*sum(ttempFedIbsS_Ibs)/len(ttempFedIbsS_Ibs),2))
                tempFedIbsS_Ci.append(round(100*sum(ttempFedIbsS_Ci)/len(ttempFedIbsS_Ci),2))
                tempFedCiS_Ibs.append(round(100*sum(ttempFedCiS_Ibs)/len(ttempFedCiS_Ibs),2))
                tempFedCiS_Ci.append(round(100*sum(ttempFedCiS_Ci)/len(ttempFedCiS_Ci),2))
                tempclient_Ibs_std.append(round(statistics.stdev(ttempclient_Ibs)*100,2))
                tempclient_Ci_std.append(round(statistics.stdev(ttempclient_Ci)*100,2))
                tempFedIbsC_Ibs_std.append(round(statistics.stdev(ttempFedIbsC_Ibs)*100,2))
                tempFedIbsC_Ci_std.append(round(statistics.stdev(ttempFedIbsC_Ci)*100,2))
                tempFedCiC_Ibs_std.append(round(statistics.stdev(ttempFedCiC_Ibs)*100,2))
                tempFedCiC_Ci_std.append(round(statistics.stdev(ttempFedCiC_Ci)*100,2))
                tempFedIbsS_Ibs_std.append(round(statistics.stdev(ttempFedIbsS_Ibs)*100,2))
                tempFedIbsS_Ci_std.append(round(statistics.stdev(ttempFedIbsS_Ci)*100,2))
                tempFedCiS_Ibs_std.append(round(statistics.stdev(ttempFedCiS_Ibs)*100,2))
                tempFedCiS_Ci_std.append(round(statistics.stdev(ttempFedCiS_Ci)*100,2))

        client_Ibs.append(tempclient_Ibs)
        client_Ci.append(tempclient_Ci)
        FedIbsC_Ibs.append(tempFedIbsC_Ibs)
        FedIbsC_Ci.append(tempFedIbsC_Ci)
        FedCiC_Ibs.append(tempFedCiC_Ibs)
        FedCiC_Ci.append(tempFedCiC_Ci)
        FedIbsS_Ibs.append(tempFedIbsS_Ibs)
        FedIbsS_Ci.append(tempFedIbsS_Ci)
        FedCiS_Ibs.append(tempFedCiS_Ibs)
        FedCiS_Ci.append(tempFedCiS_Ci)
        client_Ibs_std.append(tempclient_Ibs_std)
        client_Ci_std.append(tempclient_Ci_std)
        FedIbsC_Ibs_std.append(tempFedIbsC_Ibs_std)
        FedIbsC_Ci_std.append(tempFedIbsC_Ci_std)
        FedCiC_Ibs_std.append(tempFedCiC_Ibs_std)
        FedCiC_Ci_std.append(tempFedCiC_Ci_std)
        FedIbsS_Ibs_std.append(tempFedIbsS_Ibs_std)
        FedIbsS_Ci_std.append(tempFedIbsS_Ci_std)
        FedCiS_Ibs_std.append(tempFedCiS_Ibs_std)
        FedCiS_Ci_std.append(tempFedCiS_Ci_std)

    xarray_3d_table = xr.Dataset(
        {"Client": (("#Clients", "#Trees"), np.asarray(client_Ci))},
        coords={
            "#Clients": clients,
            "#Trees": trees,
            "Client_std": (("#Clients", "#Trees"), np.asarray(client_Ci_std)),
            "Fed_CI_Clients": (("#Clients", "#Trees"), np.asarray(FedCiC_Ci)),
            "Fed_CI_C_std": (("#Clients", "#Trees"), np.asarray(FedCiC_Ci_std)),
            "Fed_CI_Server": (("#Clients", "#Trees"), np.asarray(FedCiS_Ci)),
            "Fed_CI_Server_std": (("#Clients", "#Trees"), np.asarray(FedCiS_Ci_std)),
            "Fed_IBS_Clients": (("#Clients", "#Trees"), np.asarray(FedIbsC_Ci)),
            "Fed_IBS_C_std": (("#Clients", "#Trees"), np.asarray(FedIbsC_Ci_std)),
            "Fed_IBS_Server": (("#Clients", "#Trees"), np.asarray(FedIbsS_Ci)),
            "Fed_IBS_Server_std": (("#Clients", "#Trees"), np.asarray(FedIbsS_Ci_std)),
        },
    )
    plt.rcParams.update({'font.size': 24})
    print(xarray_3d_table.dims)
    df_table = xarray_3d_table.to_dataframe()
    print(df_table)
    #create 3D dataset
    xarray_3d = xr.Dataset(
        {"Client": (("#Clients", "#Trees"), np.asarray(client_Ci))},
        coords={
            "#Clients": clients,
            "#Trees": trees,
            "$\mathregular{Fed_{CI}Clients}$": (("#Clients", "#Trees"), np.asarray(FedCiC_Ci)),
            "$\mathregular{Fed_{CI}Server}$": (("#Clients", "#Trees"), np.asarray(FedCiS_Ci)),
            "$\mathregular{Fed_{IBS}Clients}$": (("#Clients", "#Trees"), np.asarray(FedIbsC_Ci)),
            "$\mathregular{Fed_{IBS}Server}$": (("#Clients", "#Trees"), np.asarray(FedIbsS_Ci)),
        },
    )
    df = xarray_3d.to_dataframe()
    df = df.drop(df[df.Client == 0.00].index)
    sns.set_style("darkgrid")
    bar_plot_ci = df.plot(kind='bar',figsize=(20,8), fontsize=20)
    bar_plot_ci.set_title(label=Config.data.datasource, fontdict={'fontsize':28})
    bar_plot_ci.axhline(y=baseline_ci, color= 'black', linewidth=1, linestyle='dashed', label='Baseline')
    plt.legend(loc="lower left")
    bar_plot_ci.set_ylabel("CI * 100")
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.ylim(bottom=50)
    plt.show()


    xarray_3d_table = xr.Dataset(
        {"Client": (("#Clients", "#Trees"), np.asarray(client_Ibs))},
        coords={
            "#Clients": clients,
            "#Trees": trees,
            "Client_std": (("#Clients", "#Trees"), np.asarray(client_Ibs_std)),
            "Fed_CI_Clients": (("#Clients", "#Trees"), np.asarray(FedCiC_Ibs)),
            "Fed_CI_C_std": (("#Clients", "#Trees"), np.asarray(FedCiC_Ibs_std)),
            "Fed_CI_Server": (("#Clients", "#Trees"), np.asarray(FedCiS_Ibs)),
            "Fed_CI_Server_std": (("#Clients", "#Trees"), np.asarray(FedCiS_Ibs_std)),
            "Fed_IBS_Clients": (("#Clients", "#Trees"), np.asarray(FedIbsC_Ibs)),
            "Fed_IBS_C_std": (("#Clients", "#Trees"), np.asarray(FedIbsC_Ibs_std)),
            "Fed_IBS_Server": (("#Clients", "#Trees"), np.asarray(FedIbsS_Ibs)),
            "Fed_IBS_Server_std": (("#Clients", "#Trees"), np.asarray(FedIbsS_Ibs_std)),
        },
    )
    df_table = xarray_3d_table.to_dataframe()
    print(df_table)
    #create 3D dataset
    xarray_3d = xr.Dataset(
        {"Client": (("#Clients", "#Trees"), np.asarray(client_Ibs))},
        coords={
            "#Clients": clients,
            "#Trees": trees,
            "$\mathregular{Fed_{CI}Clients}$": (("#Clients", "#Trees"), np.asarray(FedCiC_Ibs)),
            "$\mathregular{Fed_{CI}Server}$": (("#Clients", "#Trees"), np.asarray(FedCiS_Ibs)),
            "$\mathregular{Fed_{IBS}Clients}$": (("#Clients", "#Trees"), np.asarray(FedIbsC_Ibs)),
            "$\mathregular{Fed_{IBS}Server}$": (("#Clients", "#Trees"), np.asarray(FedIbsS_Ibs)),
        },
    )
    df = xarray_3d.to_dataframe()
    df = df.drop(df[df.Client == 0.00].index)
    sns.set_style("darkgrid")
    bar_plot_ibs = df.plot(kind='bar',figsize=(20,8), fontsize=20)
    bar_plot_ibs.set_title(label=Config.data.datasource, fontdict={'fontsize':28})
    bar_plot_ibs.axhline(y=baseline_ibs, color= 'black', linestyle='dashed', linewidth=1, label='Baseline')
    plt.legend(loc="lower left")
    bar_plot_ibs.set_ylabel("IBS * 100")
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.ylim(bottom=10)
    plt.show()


def make_t_test(client, baseline, Fed_IBS, Fed_CI):

    # print("t-test: client, basline")
    # print(len(client))
    # print(len(baseline))
    # t_test(client, baseline)
    print("t-test: baseline, Fed_IBS")
    t_test(baseline, Fed_IBS)
    print("t-test: baseline, Fed_CI")
    t_test(baseline, Fed_CI)
    print("t-test: Fed_CI, Fed_IBS")
    t_test_paired(Fed_CI, Fed_IBS)
    


def t_test(rvs1, rvs2):
    print(stats.ttest_ind(rvs1, rvs2))

def t_test_paired(rvs1, rvs2):
    print(stats.ttest_rel(rvs1, rvs2))

if __name__ == "__main__":
    main()