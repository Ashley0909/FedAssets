from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from functools import reduce
import numpy as np
from time import time
import random

from collections import OrderedDict, Counter
from omegaconf import DictConfig

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as ax

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

import smtplib
from email.message import EmailMessage
import ssl

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

"""some helper functions so that we can convert between numpy arrays and pytorch tensors and run our code on GPU"""
# USE_CUDA = torch.cuda.is_available() 
# USE_MPS = torch.backends.mps.is_available()

# from torch.autograd import Variable
# def cuda(v):
#     if USE_CUDA:
#         return v.cuda()
#     return v
# def toTensor(v,dtype = torch.float,requires_grad = False):
#     return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))

# def toNumpy(v):
#     if USE_CUDA:
#         return v.detach().cpu().numpy()
#     return v.detach().numpy()

# print('Using CUDA:',USE_CUDA)

malicious_record = []
benign_record = []
final_model = []
final_metric = []
e = 0
flag = 0
benign_average, malicious_average, global_targetlabel = None, None, None

class NNtrain(Strategy):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        attack_evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.attack_evaluate_fn = attack_evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn


    def __repr__(self) -> str:
        rep = f"NNtrain(accept_failures={self.accept_failures})"
        return rep


    """Return the sample size and the required number of available clients."""
    """working and unchanged"""
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients


    """Use a fraction of available clients for evaluation."""
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


    """Server request an initial global parameter given by a random client"""
    """unchanged but need changing"""
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters


    """Evaluate model parameters using an evaluation function."""
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)

        eval_res = self.evaluate_fn(server_round, parameters_ndarrays)
        if eval_res is None:
            return None
        loss, metrics = eval_res

        return loss, metrics


    """Configure the next round of training."""
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        print("Configure Fit")
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        print("Server Round:", server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_ins = FitIns(parameters, config)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


    """Configure the next round of evaluation."""
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Do not configure federated evaluation if fraction eval is 0.
        print("Configure Evaluate")
        if self.fraction_evaluate == 0.0:
            return []
    
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]


    """Original: Aggregate fit results using weighted average."""
    """Changed: Collect and combine all the results as a labelled data set"""
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        global malicious_record, final_model, final_metric, global_targetlabel, e, benign_record, benign_average, malicious_average, flag

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        """That procedure makes sure the known malicious clients are not considered"""
        weights_results = []
        all_id = []
        malicious = []
        new_results = []

        evil_parameter = []
        evil_results = []
        i = 0
        clean_total = 0
        bd_total = 0
        count = 0

        """Sample benign and malicious clients for evaluating local models"""
        local_evaluator = []
        malicious_evaluator = []
        for cp, fit_res in results:
            if fit_res.metrics["malicious"] == 0:
                local_evaluator.append(cp)
            elif fit_res.metrics["malicious"] == 1:
                malicious_evaluator.append(cp)
        
        for cp, fit_res in results:
            if cp.cid in malicious_record:
                evil_parameter.append(parameters_to_ndarrays(fit_res.parameters))
                evil_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

                """Computing local clean accuracy"""
                evaluate_ins = EvaluateIns(fit_res.parameters, {})
                evaluator = random.choice(local_evaluator)
                evaluate_res = evaluator.evaluate(evaluate_ins, None)
                clean_total += evaluate_res.metrics["accuracy"]
                count += 1

                """Computing local backdoor accuracy"""
                bd_evaluator = random.choice(malicious_evaluator)
                bd_res = bd_evaluator.evaluate(evaluate_ins, None)
                bd_total += bd_res.metrics["global_poison"]
            else:
                all_id.append((i, cp.cid))
                malicious.append(str(fit_res.metrics["malicious"]))
                weights_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
                new_results.append((cp, fit_res))
                i += 1

        print("malicious is", malicious)
        
        num_examples = [res[1] for res in weights_results]
        print("Number of Evil Clients:", len(evil_results))

        if len(num_examples) == 0:
            print("Only Evil Clients in this round, void this round.")
            return final_model, final_metric
        
        client_id = np.array(all_id)[:,1]
        local_cid = np.array(all_id)[:,0]
        parameter = [client[0] for client in weights_results]  #[c1:[10array], c2:[10array], ...]  #client[1] is the number of examples

        print("Number of Preset Malicious Clients is", malicious.count("1"))
        print("Number of Preset Benign Clients is", malicious.count("0"))

        """Computing local accuracies of this round's attackers"""
        for x in range(len(new_results)):
            if malicious[x] == '1':
                _, fit_res = new_results[x]
                evaluate_ins = EvaluateIns(fit_res.parameters, {})
                evaluator = random.choice(local_evaluator)
                evaluate_res = evaluator.evaluate(evaluate_ins, None)
                clean_total += evaluate_res.metrics["accuracy"]

                bd_evaluator = random.choice(malicious_evaluator)
                bd_res = bd_evaluator.evaluate(evaluate_ins, None)
                bd_total += bd_res.metrics["global_poison"]

        if malicious.count("1") != 0 or count > 0:
            clean_acc = clean_total/(malicious.count("1")+count)
            bd_acc = bd_total/(malicious.count("1")+count)
            print("Local Clean Data Accuracy:", clean_acc)
            print("Local Backdoor accuracy:", bd_acc)
        else:
            clean_acc = "N/A"
            bd_acc = "N/A"
            print("Local Clean accuracy: N/A")
            print("Local Backdoor accuracy: N/A")

        """Get FC Weight for clustering"""
        fcw = []
        for i in range(len(parameter)): 
            w = 0
            vector = []  #set up a vector for each client
            for j in range(len(parameter[i][-2])): #10
                w = np.sum(parameter[i][-2][j]) #84
                vector.append(w)
            fcw.append(np.array(vector))

        if len(evil_results) > 0:
            evil_fcw = []
            for i in range(len(evil_results)): 
                w = 0
                vector = []  #set up a vector for each client
                for j in range(len(evil_parameter[i][-2])): #10
                    w = np.sum(evil_parameter[i][-2][j]) #84
                    vector.append(w)
                evil_fcw.append(np.array(vector))
        else:
            evil_fcw = []

        comb_C, e, flag = nd_clustering(parameter, local_cid, malicious, fcw, "50C15", server_round, e, flag, 2)

        """Assume Clustering 100%"""
        comb_C = np.array(malicious).astype(int)
        print("comb_C is", comb_C)

        # heatmaps(local_cid, comb_C, evil_fcw, np.array(fcw), 'Output Layer Weights', server_round)

        record, acc_diff = 1, 0
        
        if benign_average == None and malicious_average == None:  # KMeans and 2 clusters (First round)
            """Allocate good and bad clients"""
            good_clients = local_cid[comb_C == 0]
            bad_clients = local_cid[comb_C == 1]

            """Split the parameters into good and malicious"""
            good_fcw = np.array(fcw)[comb_C == 0]
            bad_fcw = np.array(fcw)[comb_C == 1]

            """Detecting Target Label in the first round"""
            if (len(good_clients) > 0) and (len(bad_clients) > 0 or len(evil_results) > 0):
                dist_list, sign_list, vardist_list = [], [], []
                good_averages, bad_averages = [], []
                for i in range(len(good_fcw[0])):
                    good_biases_average = compute_average(good_fcw[:,i], len(good_clients))
                    good_biases_variance = np.var(good_fcw[:,i])
                    if len(bad_clients) == 0:
                        bad_biases_average = compute_average(np.array(evil_fcw)[:,i], len(evil_results))
                        bad_biases_variance = np.var(evil_fcw[:,i])
                    elif len(evil_results) == 0:
                        bad_biases_average = compute_average(bad_fcw[:,i], len(bad_clients))
                        bad_biases_variance = np.var(bad_fcw[:,i])
                    else:
                        bad_biases_average = compute_average(np.concatenate((bad_fcw, evil_fcw), axis=0)[:,i], (len(bad_clients)+len(evil_results)))
                        bad_biases_variance = np.var(np.concatenate((bad_fcw, evil_fcw), axis=0)[:,i])

                    var_dist = abs(good_biases_variance - bad_biases_variance)
                    dist = abs(good_biases_average - bad_biases_average)
                    sign = np.sign(good_biases_average - bad_biases_average)
                    sign_list.append(sign)
                    dist_list.append(dist)
                    vardist_list.append(var_dist)
                    bad_averages.append(bad_biases_average)
                    good_averages.append(good_biases_average)

                # Determine the target label (considering the case of wrong cluster and counterexample)
                arr_list = np.array(dist_list)
                max_label = np.argmax(arr_list)
                arr_copy = arr_list.copy()
                arr_copy[max_label] = -np.inf
                # second_max_label = np.argmax(arr_copy)
                # if abs(arr_list[max_label] - arr_list[second_max_label]) < 1.0:
                #     target_label = np.argmax(vardist_list)
                # else:
                target_label = np.argmax(arr_list)

                if sign_list[target_label] == 1:
                    comb_C = np.array([1 if x == 0 else 0 if x == 1 else x for x in comb_C])
                    benign_average = bad_averages[target_label]
                    malicious_average = good_averages[target_label]
                else:
                    benign_average = good_averages[target_label]
                    malicious_average = bad_averages[target_label]
                global_targetlabel = target_label
                print("Target label is", global_targetlabel)
                record = 0
                acc_diff = 0
        else:
            comb_C, record, acc_diff = merge_clients(comb_C, fcw, local_cid, benign_average, malicious_average, global_targetlabel)
            """Assume Clustering 100%"""
            comb_C = np.array(malicious).astype(int)

        print("Now, comb_C is", comb_C)

        """Compute accuracies"""
        correct = 0
        for i in range(len(comb_C)):
            if (comb_C[i] == 0) and (malicious[i] == "0"):
                correct += 1
            elif (comb_C[i] == 1) and (malicious[i] == "1"):
                correct += 1
        clustering_acc = correct / len(malicious)

        print("Clustering Accuracy is", clustering_acc)

        if record == 1:
            global_bad = client_id[comb_C == 1]
            malicious_record.extend(global_bad)
            malicious_record = list(set(malicious_record)) #avoid duplicates

            global_good = client_id[comb_C == 0]
            benign_record.extend(global_good)
            benign_record = list(set(benign_record))

        """After detecting the clients and their target label, make a function that determines the weight of contribution"""
        good_results = [weights_results[i] for i in range(len(weights_results)) if comb_C[i] == 0]
        bad_results = [weights_results[i] for i in range(len(weights_results)) if comb_C[i] == 1]
        print("length of good results is", len(good_results), "and length of bad results is", len(bad_results))

        parameters_aggregated = ndarrays_to_parameters(resnet_aggregate(good_results, bad_results, evil_results, acc_diff, global_targetlabel))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Record the final model in case if the next round is a void round
        final_model = parameters_aggregated
        final_metric = metrics_aggregated

        return parameters_aggregated, metrics_aggregated

    """Aggregate evaluation losses using weighted average."""
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        print("Aggregate evaluate")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        """========Accuracy on Clean Data (Main)========"""
        valid_results = []
        print("benign record is", benign_record)
        if benign_record != []:
            for cp, evaluate_res in results:
                
                if cp.cid in benign_record:
                    valid_results.append((evaluate_res.num_examples, evaluate_res.loss))

        if valid_results != []:
            loss_aggregated = weighted_loss_avg(valid_results)
        else:
            print("No valid result")
            loss_aggregated = weighted_loss_avg(
                [
                    (evaluate_res.num_examples, evaluate_res.loss)
                    for _, evaluate_res in results
                ]
            )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = []
            if benign_record != []:
                for cp, res in results:
                    if cp.cid in benign_record:
                        eval_metrics.append((res.num_examples, res.metrics))
            if eval_metrics == []:
                eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        if server_round > 0:
            print("Federated Accuracy on Clean Data:", metrics_aggregated["accuracy"])

        """========Accuracy on Dirty Data========"""
        # Aggregate custom metrics if aggregation fn was provided
        dirty_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            dirty_metrics = []
            if malicious_record != []:
                for cp, res in results:
                    if cp.cid in malicious_record:
                        dirty_metrics.append((res.num_examples, res.metrics))
            if dirty_metrics == []:
                dirty_metrics = [(res.num_examples, res.metrics) for _, res in results]
            dirty_aggregated = self.evaluate_metrics_aggregation_fn(dirty_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        if server_round > 0:
            print("Federated Accuracy on Dirty Data:", dirty_aggregated["accuracy"])
            print("Global Poisoning Accuracy on Dirty Data:", dirty_aggregated["global_poison"])

        if server_round == 100:
            email_sender = '09auhoiting@gmail.com'
            email_password = 'fgkkhdwmluvpxewp'
            email_receiver = '09auhoiting@gmail.com'
            subject = 'Vscode Run Result'
            body = 'Ran Successfully. Final Accuracy is {GA}'.format(GA=metrics_aggregated["accuracy"])

            em = EmailMessage()
            em['From'] = email_sender
            em['To'] = email_receiver
            em['Subject'] = subject
            em.set_content(body)

            context = ssl.create_default_context()

            with smtplib.SMTP_SSL('smtp.gmail.com',465, context=context) as smtp:
                smtp.login(email_sender, email_password)
                smtp.sendmail(email_sender, email_receiver, em.as_string())

        return loss_aggregated, metrics_aggregated

def nd_clustering(parameter, cid, malicious, layer, name, server_round, e, flag, dim):
    label = []
    textstr = ''
    for k in range(len(parameter)):
        label.append("Client" + str(cid[k]) + "\u2192" + malicious[k] + "\n")
        textstr += f'Client {str(cid[k])} \u2192 {malicious[k]} \n'
    
    if len(parameter) < 2:
        return [-1], e, flag
        
    """Run PCA on the n dimensional data for the first round"""
    pca = PCA(n_components=dim)
    reduced_data = pca.fit_transform(layer)
    
    if flag == 0:
        kmeans = KMeans(init="k-means++", n_clusters=2, n_init=4).fit(reduced_data)
        comb_C = kmeans.predict(reduced_data)

        centroids = kmeans.cluster_centers_
        centroids_assignment = kmeans.predict(centroids)
        unique_labels = np.unique(comb_C)
        # Compute intra-cluster distances
        intra_cluster = euclidean_distances(centroids[centroids_assignment == unique_labels[0]], centroids[centroids_assignment == unique_labels[1]])
        if intra_cluster[0][0] < 0.05:
            flag = 1

        max_dist = []
        for l in unique_labels:
            distances = euclidean_distances(reduced_data[comb_C == l], centroids[centroids_assignment == l])
            max_dist.append(np.max(distances))
        e = np.max(max_dist)
    else:
        mp = 5
        e -= 0.0025*(server_round/5)
        mp -= (server_round//20) 
        db = DBSCAN(eps=max(e, 0.03), min_samples=max(3,mp)).fit(reduced_data)
        comb_C = db.labels_

    unique_labels = np.unique(comb_C)

    """Set the largest cluster to be benign as a default"""
    counts = Counter(comb_C)
    benign_class = max(counts, key=counts.get)

    # Update comb_C so that benign is 0 and malicious is 1 (For only 2 clusters, and only 1 cluster (assume all to be good))
    if len(unique_labels) < 3:
        mod_combC = [0 if item == benign_class else 1 for item in comb_C]
        comb_C = np.array(mod_combC)

    return comb_C, e, flag

def heatmaps(local_cid, comb_C, evil_layer, layer, name, server_round):
    textstr = ''
    good_client = local_cid[comb_C == 0]
    bad_client = local_cid[comb_C == 1]

    for i in range(len(good_client)):
        textstr += f'Client {str(good_client[i])} \u2192 0 \n'
    for i in range(len(bad_client)):
        textstr += f'Client {str(bad_client[i])} \u2192 1 \n'

    good_layer = layer[comb_C == 0]
    bad_layer = layer[comb_C == 1]

    if evil_layer != []:
        plt.imshow(np.concatenate((good_layer, bad_layer, np.array(evil_layer)), axis=0), cmap='viridis', interpolation='nearest')
    else:
        plt.imshow(np.concatenate((good_layer, bad_layer), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.annotate(textstr, xy=(0,0.5), verticalalignment='center',  horizontalalignment='left', xycoords='figure fraction')
    plt.xlabel(name)
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' {0} in Round {1}".format(name, server_round))
    plt.savefig('Round {0} {1} Non-IID.png'.format(server_round, name))
    plt.close()

def compute_average(data, count):
    average = np.sum(data, axis=0) / count
    return average

def resnet_aggregate(good_result, bad_result, evil_result, acc_diff, target_label):
    malicious_result = bad_result + evil_result

    good_numex_total = sum([num_examples for _, num_examples in good_result])
    malicious_numex_total = sum([num_examples for _, num_examples in malicious_result])

    # Create a list of weights, each multiplied by the related number of examples
    good_weighted_param = [[layer * num_examples for layer in weights] for weights, num_examples in good_result]

    good_prime: NDArrays = [
        reduce(np.add, layer_updates) / good_numex_total
        for layer_updates in zip(*good_weighted_param)
    ]

    """All Benign"""
    # return good_prime

    mali_weighted_params = [[layer * num_examples for layer in weights] for weights, num_examples in malicious_result]

    malicious_prime: NDArrays = [
        reduce(np.add, layer_updates) / malicious_numex_total
        for layer_updates in zip(*mali_weighted_params)
    ]

    # For each layer, compute the distance between the good and the bad, then apply the similarity weight to the bad clients
    # Depending on the layer being weight or bias, we have different approaches: directly calc dist for bias since one value per neuron, but sum all weights of a neuron before calc dist
    aggregated_result = []
    if acc_diff == 0:
        for l in range(len(good_prime)-2, len(good_prime)):  # Number of fully connected layer
            if len(good_prime[l].shape) > 1: # weights: sum up all incoming weights
                if malicious_prime != []:
                    neuron_dists = list(map(abs, map(lambda x,y: x - y, [sum(x) for x in good_prime[l]], [sum(y) for y in malicious_prime[l]])))
                    # sim_weight_list = [np.exp(-13 * neuron_dists[i]) for i in range(len(neuron_dists))]  #EQ
                    sim_weight_list = [np.exp(-13 * neuron_dists[i]) if i != target_label else 0 for i in range(len(neuron_dists))]  #Mine
                    weighted_param_aggregated = [
                        [
                            (g+(s*b)) / (1+s)
                            for g,b, in zip(good_sublist, bad_sublist)
                        ]
                        for good_sublist, bad_sublist, s in zip(good_prime[l], malicious_prime[l], sim_weight_list)
                    ]
                else:
                    weighted_param_aggregated = good_prime[l]
            elif len(good_prime[l].shape) < 1:
                if malicious_prime != []:
                    dist = abs(good_prime[l] - malicious_prime[l])
                    sim_weight = np.exp(-13 * dist)
                    weighted_param_aggregated = (good_prime[l] + (sim_weight * malicious_prime[l]))/(1+sim_weight)
                else:
                    weighted_param_aggregated = good_prime[l]
            else:
                if malicious_prime != []:
                    neuron_dists = list(map(abs, map(lambda x,y: x - y, good_prime[l], malicious_prime[l])))
                    # sim_weight_list = [np.exp(-13 * neuron_dists[i]) for i in range(len(neuron_dists))]  #EQ
                    sim_weight_list = [np.exp(-13 * neuron_dists[i]) if i != target_label else 0 for i in range(len(neuron_dists))]  #Mine
                    weighted_param_aggregated = list(map(lambda g,b,s: (g + (s * b))/ (1 + s), good_prime[l], malicious_prime[l], sim_weight_list)) 
                else:
                    weighted_param_aggregated = good_prime[l]

            aggregated_result.append(weighted_param_aggregated)  # records each layer
        
        good_prime[-2:] = aggregated_result[:2]  # Number of fully connected layer

    # Still need to consider the case when there is no good client
    elif malicious_prime != []:
        sim_weight = np.exp(-13 * acc_diff)
        for l in range(len(malicious_prime)):
            weighted_param_aggregated = (sim_weight * malicious_prime[l])/sim_weight
            good_prime.append(weighted_param_aggregated)  # records each layer

    return good_prime

def merge_clients(comb_C, fcw, local_cid, benign_average, malicious_average, global_targetlabel):
    unique_labels = np.unique(comb_C)
    client_status = {elem: 1 for elem in unique_labels}  # set all clusters to be malicious first
    all_malicious = 1
    record = 1
    acc_diff = 0

    for l in unique_labels:
        c_fcw = np.array(fcw)[comb_C == l]
        c_ids = local_cid[comb_C == l]
        c_average = compute_average(c_fcw[:,global_targetlabel], len(c_ids))
        if abs(malicious_average - c_average) > abs(benign_average - c_average):  #the status should be benign
            client_status[l] = 0
            all_malicious = 0

    if len(unique_labels) == 1:
        record = 0

    if all_malicious == 1: # if there is no benign clients, we need to find acc_diff
        acc_diff = abs(benign_average - c_average)
        
    mod_combC = [1 if client_status[l] == 1 else 0 for l in comb_C]
    comb_C = np.array(mod_combC)

    return comb_C, record, acc_diff