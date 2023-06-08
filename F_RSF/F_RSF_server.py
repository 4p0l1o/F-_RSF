"""
A simple federated learning server using federated averaging.
"""

import asyncio
import logging
import os
import pickle
import sys

import utils as utils

from plato.algorithms import registry as algorithms_registry
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import all_inclusive
from plato.servers import base
from plato.trainers import registry as trainers_registry
from plato.utils import csv_processor, fonts

class Server(base.Server):
    """Federated learning server using federated averaging."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(callbacks=callbacks)

        self.custom_model = model
        self.model = None
        self.saveFile = None

        self.custom_algorithm = algorithm
        self.algorithm = None

        self.custom_trainer = trainer
        self.trainer = None

        self.custom_datasource = datasource
        self.datasource = None

        self.testset = None
        self.testset_sampler = None
        self.trainset = None
        self.total_samples = 0

        self.total_clients = Config().clients.total_clients
        self.clients_per_round = Config().clients.per_round

        logging.info(
            "[Server #%d] Started training on %d clients with %d per round.",
            os.getpid(),
            self.total_clients,
            self.clients_per_round,
        )

    def configure(self) -> None:
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        super().configure()

        total_rounds = Config().trainer.rounds
        target_accuracy = None
        target_perplexity = None

        if hasattr(Config().trainer, "target_accuracy"):
            target_accuracy = Config().trainer.target_accuracy
        elif hasattr(Config().trainer, "target_perplexity"):
            target_perplexity = Config().trainer.target_perplexity

        if target_accuracy:
            logging.info(
                "Training: %s rounds or accuracy above %.1f%%\n",
                total_rounds,
                100 * target_accuracy,
            )
        elif target_perplexity:
            logging.info(
                "Training: %s rounds or perplexity below %.1f\n",
                total_rounds,
                target_perplexity,
            )
        else:
            logging.info("Training: %s rounds\n", total_rounds)

        self.init_trainer()

        # Prepares this server for processors that processes outbound and inbound
        # data payloads
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Server", server_id=os.getpid(), trainer=self.trainer
        )

        if not (hasattr(Config().server, "do_test") and not Config().server.do_test):
            if self.datasource is None and self.custom_datasource is None:
                self.datasource = datasources_registry.get(client_id=0)
            elif self.datasource is None and self.custom_datasource is not None:
                self.datasource = self.custom_datasource()

            self.testset = self.datasource.get_test_set()
            self.trainset = self.datasource.get_train_set()
            if hasattr(Config().data, "testset_size"):
                self.testset_sampler = all_inclusive.Sampler(
                    self.datasource, testing=True
                )

        # Initialize the test accuracy csv file if clients compute locally
        if hasattr(Config().clients, "do_test") and Config().clients.do_test:
            folder = (
                f"{Config().params['result_path']}/{Config.data.datasource}"
                )
            if not os.path.exists(folder):
                # If it doesn't exist, create it
                os.makedirs(folder)

            baseLinePath = (
                f"{Config().params['result_path']}/{Config.data.datasource}/Baseline_accuracy.csv"
            )
            accuracy_headers = ["client_id", "CI", "IBS"]
            if not os.path.exists(baseLinePath):
                csv_processor.initialize_csv(
                    baseLinePath, accuracy_headers, Config().params["result_path"]
                )
            self.saveFile = (
                f"{Config().params['result_path']}/{Config.data.datasource}/accuracy_{Config.clients.total_clients}_clients_{int(Config.server.n_trees/Config.clients.total_clients)}_trees.csv"
            )
            if not os.path.exists(self.saveFile):
                csv_processor.initialize_csv(
                    self.saveFile, accuracy_headers, Config().params["result_path"]
                )

            parameters = ["client_id", "n_estimators", "min_samples_split", "min_samples_leaf", "max_features", "max_depth", "bootstrap"]
            path = (
                f"{Config().params['result_path']}/{Config.data.datasource}/params.csv"
            )
            if not os.path.exists(path):
                csv_processor.initialize_csv(
                    path, parameters, Config().params["result_path"]
                )
            path= (
                f"{Config().params['result_path']}/{Config.data.datasource}/params_{Config.clients.total_clients}_clients.csv"
            )
            if not os.path.exists(path):
                csv_processor.initialize_csv(
                    path, parameters, Config().params["result_path"]
                )

        if Config.server.do_baseline_test:
            weights = self.trainer.train(self.trainset, "baseline", "")
            baseline_score = self.trainer.test(self.testset, self.trainset, weights)
            logging.info(
                fonts.colourize(
                    f"[{self}] Baseline model c-index: {100 * baseline_score[0]:.2f}%\n"
                )
            )
            accuracy_row = [
                "baseline",
                baseline_score[0],
                baseline_score[1]
            ]
            csv_processor.write_csv(baseLinePath, accuracy_row)

    def init_trainer(self) -> None:
        """Setting up the global model, trainer, and algorithm."""
        if self.model is None and self.custom_model is not None:
            self.model = self.custom_model

        if self.trainer is None and self.custom_trainer is None:
            self.trainer = trainers_registry.get(model=self.model)
        elif self.trainer is None and self.custom_trainer is not None:
            self.trainer = self.custom_trainer(model=self.model)

        if self.algorithm is None and self.custom_algorithm is None:
            self.algorithm = algorithms_registry.get(trainer=self.trainer)
        elif self.algorithm is None and self.custom_algorithm is not None:
            self.algorithm = self.custom_algorithm(trainer=self.trainer)

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)
        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update
    
    def aggregate_weights(self, updates, baseline_weights, weights_received):

        print("aggregating weights!")
        return self.algorithm.update_weights(weights_received)

    async def _process_reports(self):
        """Process the client reports by aggregating their weights."""
        weights_received = [update.payload for update in self.updates]
        weights_received = self.weights_received(weights_received)
        self.callback_handler.call_event("on_weights_received", self, weights_received)

        # Extract the current model weights as the baseline
        baseline_weights = self.algorithm.extract_weights()
        updated_weights = []

        if hasattr(self, "aggregate_weights"):
            # Runs a server aggregation algorithm using weights rather than deltas
            logging.info(
                "[Server #%d] Aggregating model weights directly rather than weight deltas.",
                os.getpid(),
            )
            updated_weights = self.aggregate_weights(
                self.updates, baseline_weights, weights_received
            )
        else:
            # Computes the weight deltas by comparing the weights received with
            # the current global model weights
            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights, weights_received
            )
            # Runs a framework-agnostic server aggregation algorithm, such as
            # the federated averaging algorithm
            logging.info("[Server #%d] Aggregating model weight deltas.", os.getpid())
            deltas = await self.aggregate_deltas(self.updates, deltas_received)
            # Updates the existing model weights from the provided deltas
            updated_weights = self.algorithm.update_weights(deltas)
            # Loads the new model weights
            self.algorithm.load_weights(updated_weights)

        # The model weights have already been aggregated, now calls the
        # corresponding hook and callback
        self.weights_aggregated(self.updates)
        self.callback_handler.call_event("on_weights_aggregated", self, self.updates)

        # Testing the global model accuracy
        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                "[%s] Average client accuracy: %.2f%%.", self, 100 * self.accuracy
            )
        else:
            if self.current_round == Config.trainer.rounds:
                datasets = self.datasource.get_all_clients_dataset()
                ibs_weights = [updated_weights["ibs"], updated_weights["feature_names"], updated_weights["event_times"], updated_weights["output"], "Fed_IBS"]
                ci_weights = [updated_weights["ci"], updated_weights["feature_names"], updated_weights["event_times"], updated_weights["output"], "FED_CI"]
                
                self.trainer.test_all_clients(datasets, ibs_weights)
                self.trainer.test_all_clients(datasets, ci_weights)

                # Testing the updated model directly at the server
                logging.info("[%s] Started model testing.", self)
                ibs_accuracy = self.trainer.test(self.testset, self.trainset, ibs_weights, self.testset_sampler)
                print(len(updated_weights))
                accuracy_row = [
                    f"Federated_ibs_S",
                    ibs_accuracy[0],
                    ibs_accuracy[1]
                ]
                csv_processor.write_csv(self.saveFile, accuracy_row)

                ci_accuracy = self.trainer.test(self.testset, self.trainset, ci_weights, self.testset_sampler)
                print(len(updated_weights))
                accuracy_row = [
                    f"Federated_ci_S",
                    ci_accuracy[0],
                    ci_accuracy[1]
                ]
                csv_processor.write_csv(self.saveFile, accuracy_row)

                logging.info(
                    fonts.colourize(
                        f"[{self}] Global ibs model c-index: {100 * ibs_accuracy[0]:.2f}%\n"
                    )
                )
                logging.info(
                    fonts.colourize(
                        f"[{self}] Global ibs model ibs: {100 * ibs_accuracy[1]:.2f}%\n"
                    )
                )
                logging.info(
                    fonts.colourize(
                        f"[{self}] Global ci model c-index: {100 * ci_accuracy[0]:.2f}%\n"
                    )
                )
                logging.info(
                    fonts.colourize(
                        f"[{self}] Global ci model ibs: {100 * ci_accuracy[1]:.2f}%\n"
                    )
                )

        self.clients_processed()
        self.callback_handler.call_event("on_clients_processed", self)

    def clients_processed(self) -> None:
        """Additional work to be performed after client reports have been processed."""
        print("something to do here!")

    def get_logged_items(self) -> dict:
        """Get items to be logged by the LogProgressCallback class in a .csv file."""
        return {
            "round": self.current_round,
            "accuracy": self.accuracy,
            "elapsed_time": self.wall_time - self.initial_wall_time,
            "comm_time": max(update.report.comm_time for update in self.updates),
            "round_time": max(
                update.report.training_time + update.report.comm_time
                for update in self.updates
            ),
            "comm_overhead": self.comm_overhead,
        }

    @staticmethod
    def accuracy_averaging(updates):
        """Compute the average accuracy across clients."""
        # Get total number of samples
        total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        accuracy = 0
        for update in updates:
            accuracy += update.report.accuracy * (
                update.report.num_samples / total_samples
            )

        return accuracy

    def weights_received(self, weights_received):
        """
        Method called after the updated weights have been received.
        """
        return weights_received

    def weights_aggregated(self, updates):
        """
        Method called after the updated weights have been aggregated.
        """

    def choose_clients(self, clients_pool, clients_count):
        """Chooses a subset of the clients to participate in each round."""
        assert clients_count <= len(clients_pool)
        selected_clients = []
        for i in range(((self.current_round-1)*Config.clients.per_round)+1, ((self.current_round-1)*Config.clients.per_round)+Config.clients.per_round+1):
            print(i)
            selected_clients.append(i)

        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients

    async def _select_clients(self, for_next_batch=False):
        """Selects a subset of the clients and send messages to them to start training."""
        if not for_next_batch:
            self.updates = []
            self.current_round += 1
            self.round_start_wall_time = self.wall_time

            if hasattr(Config().trainer, "max_concurrency"):
                self.trained_clients = []

            logging.info(
                fonts.colourize(
                    f"\n[{self}] Starting round {self.current_round}/{Config().trainer.rounds}."
                )
            )

            if Config().is_central_server():
                # In cross-silo FL, the central server selects from the pool of edge servers
                self.clients_pool = list(self.clients)

            elif not Config().is_edge_server():
                self.clients_pool = list(range(1, 1 + self.total_clients))

            # In asychronous FL, avoid selecting new clients to replace those that are still
            # training at this time

            # When simulating the wall clock time, if len(self.reported_clients) is 0, the
            # server has aggregated all reporting clients already
            if (
                self.asynchronous_mode
                and self.selected_clients is not None
                and len(self.reported_clients) > 0
                and len(self.reported_clients) < self.clients_per_round
            ):
                # If self.selected_clients is None, it implies that it is the first iteration;
                # If len(self.reported_clients) == self.clients_per_round, it implies that
                # all selected clients have already reported.

                # Except for these two cases, we need to exclude the clients who are still
                # training.
                training_client_ids = [
                    self.training_clients[client_id]["id"]
                    for client_id in self.training_clients
                ]

                # If the server is simulating the wall clock time, some of the clients who
                # reported may not have been aggregated; they should be excluded from the next
                # round of client selection
                reporting_client_ids = [
                    client[2]["client_id"] for client in self.reported_clients
                ]

                selectable_clients = [
                    client
                    for client in self.clients_pool
                    if client not in training_client_ids
                    and client not in reporting_client_ids
                ]

                if self.simulate_wall_time:
                    self.selected_clients = self.choose_clients(
                        selectable_clients, len(self.current_processed_clients)
                    )
                else:
                    self.selected_clients = self.choose_clients(
                        selectable_clients, len(self.reported_clients)
                    )
            else:
                self.selected_clients = self.choose_clients(
                    self.clients_pool, self.clients_per_round
                )

            self.current_reported_clients = {}
            self.current_processed_clients = {}

            # There is no need to clear the list of reporting clients if we are
            # simulating the wall clock time on the server. This is because
            # when wall clock time is simulated, the server needs to wait for
            # all the clients to report before selecting a subset of clients for
            # replacement, and all remaining reporting clients will be processed
            # in the next round
            if not self.simulate_wall_time:
                self.reported_clients = []

        if len(self.selected_clients) > 0:
            self.selected_sids = []

            # If max_concurrency is specified, run selected clients batch by batch,
            # and the number of clients in each batch (on each GPU, if multiple GPUs are available)
            # is equal to # (or maybe smaller than for the last batch) max_concurrency
            if (
                hasattr(Config().trainer, "max_concurrency")
                and not Config().is_central_server()
            ):
                selected_clients = []
                if Config().gpu_count() > 1:
                    untrained_clients = list(
                        set(self.selected_clients).difference(self.trained_clients)
                    )
                    available_gpus = Config().gpu_count()
                    for cuda_id in range(available_gpus):
                        for client_id in untrained_clients:
                            if client_id % available_gpus == cuda_id:
                                selected_clients.append(client_id)
                            if len(selected_clients) >= min(
                                len(self.clients),
                                (cuda_id + 1) * Config().trainer.max_concurrency,
                                self.clients_per_round,
                            ):
                                break
                        # There is no enough alive clients, break the selection
                        if len(selected_clients) >= len(self.clients):
                            break
                else:
                    selected_clients = self.selected_clients[
                        len(self.trained_clients) : min(
                            len(self.trained_clients) + len(self.clients),
                            len(self.selected_clients),
                        )
                    ]

                self.trained_clients += selected_clients

            else:
                selected_clients = self.selected_clients

            for selected_client_id in selected_clients:
                self.selected_client_id = selected_client_id

                if Config().is_central_server():
                    client_process_id = selected_client_id
                else:
                    client_processes = [client for client in self.clients]

                    # Find a client process that is currently not training
                    # or selected in this round
                    for process_id in client_processes:
                        current_sid = self.clients[process_id]["sid"]
                        if not (
                            current_sid in self.training_sids
                            or current_sid in self.selected_sids
                        ):
                            client_process_id = process_id
                            break

                sid = self.clients[client_process_id]["sid"]

                # Track the selected client process
                self.training_sids.append(sid)
                self.selected_sids.append(sid)

                # Assign the client id to the client process
                self.clients[client_process_id]["client_id"] = self.selected_client_id

                self.training_clients[self.selected_client_id] = {
                    "id": self.selected_client_id,
                    "starting_round": self.current_round,
                    "start_time": self.round_start_wall_time,
                    "update_requested": False,
                }

                logging.info(
                    "[%s] Selecting client #%d for training.",
                    self,
                    self.selected_client_id,
                )

                server_response = {
                    "id": self.selected_client_id,
                    "current_round": self.current_round,
                }
                server_response = self.customize_server_response(
                    server_response, client_id=self.selected_client_id
                )
                payload = self.algorithm.extract_weights()
                #payload = self.customize_server_payload(payload)
                if self.comm_simulation:
                    logging.info(
                        "[%s] Sending the current model to client #%d (simulated).",
                        self,
                        self.selected_client_id,
                    )

                    # First apply outbound processors, if any
                    payload = self.outbound_processor.process(payload)

                    model_name = (
                        Config().trainer.model_name
                        if hasattr(Config().trainer, "model_name")
                        else "custom"
                    )
                    checkpoint_path = Config().params["checkpoint_path"]

                    payload_filename = (
                        f"{checkpoint_path}/{model_name}_{self.selected_client_id}.pth"
                    )

                    with open(payload_filename, "wb") as payload_file:
                        pickle.dump(payload, payload_file)

                    server_response["payload_filename"] = payload_filename

                    payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2

                    logging.info(
                        "[%s] Sending %.2f MB of payload data to client #%d (simulated).",
                        self,
                        payload_size,
                        self.selected_client_id,
                    )

                    self.comm_overhead += payload_size

                    # Compute the communication time to transfer the current global model to client
                    self.downlink_comm_time[self.selected_client_id] = payload_size / (
                        (self.downlink_bandwidth / 8) / len(self.selected_clients)
                    )

                # Send the server response as metadata to the clients (payload to follow)
                await self.sio.emit(
                    "payload_to_arrive", {"response": server_response}, room=sid
                )

                if not self.comm_simulation:
                    # Send the server payload to the client
                    logging.info(
                        "[%s] Sending the current model to client #%d.",
                        self,
                        selected_client_id,
                    )

                    await self._send(sid, payload, selected_client_id)

            self.clients_selected(self.selected_clients)
            self.callback_handler.call_event(
                "on_clients_selected", self, self.selected_clients
            )
