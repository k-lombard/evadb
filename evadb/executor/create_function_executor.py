# coding=utf-8
# Copyright 2018-2023 EvaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import hashlib
import locale
import os
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from evadb.catalog.catalog_utils import get_metadata_properties
from evadb.catalog.models.function_catalog import FunctionCatalogEntry
from evadb.catalog.models.function_io_catalog import FunctionIOCatalogEntry
from evadb.catalog.models.function_metadata_catalog import FunctionMetadataCatalogEntry
from evadb.configuration.constants import (
    DEFAULT_TRAIN_REGRESSION_METRIC,
    DEFAULT_TRAIN_TIME_LIMIT,
    DEFAULT_XGBOOST_TASK,
    EvaDB_INSTALLATION_DIR,
)
from evadb.database import EvaDBDatabase
from evadb.executor.abstract_executor import AbstractExecutor
from evadb.functions.decorators.utils import load_io_from_function_decorators
from evadb.models.storage.batch import Batch
from evadb.plan_nodes.create_function_plan import CreateFunctionPlan
from evadb.third_party.huggingface.create import gen_hf_io_catalog_entries
from evadb.utils.errors import FunctionIODefinitionError
from evadb.utils.generic_utils import (
    load_function_class_from_file,
    string_comparison_case_insensitive,
    try_to_import_ludwig,
    try_to_import_neuralforecast,
    try_to_import_sklearn,
    try_to_import_statsforecast,
    try_to_import_torch,
    try_to_import_ultralytics,
    try_to_import_xgboost,
)
from evadb.utils.logging_manager import logger


# From https://stackoverflow.com/a/34333710
@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.

    >>> with set_env(PLUGINS_DIR='test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    :type environ: dict[str, unicode]
    :param environ: Environment variables to set
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


class CreateFunctionExecutor(AbstractExecutor):
    def __init__(self, db: EvaDBDatabase, node: CreateFunctionPlan):
        super().__init__(db, node)
        self.function_dir = Path(EvaDB_INSTALLATION_DIR) / "functions"

    def handle_huggingface_function(self):
        """Handle HuggingFace functions

        HuggingFace functions are special functions that are not loaded from a file.
        So we do not need to call the setup method on them like we do for other functions.
        """
        # We need at least one deep learning framework for HuggingFace
        # Torch or Tensorflow
        try_to_import_torch()
        impl_path = f"{self.function_dir}/abstract/hf_abstract_function.py"
        io_list = gen_hf_io_catalog_entries(self.node.name, self.node.metadata)
        return (
            self.node.name,
            impl_path,
            self.node.function_type,
            io_list,
            self.node.metadata,
        )

    def handle_ludwig_function(self):
        """Handle ludwig functions

        Use Ludwig's auto_train engine to train/tune models.
        """
        try_to_import_ludwig()
        from ludwig.automl import auto_train

        assert (
            len(self.children) == 1
        ), "Create ludwig function expects 1 child, finds {}.".format(
            len(self.children)
        )

        aggregated_batch_list = []
        child = self.children[0]
        for batch in child.exec():
            aggregated_batch_list.append(batch)
        aggregated_batch = Batch.concat(aggregated_batch_list, copy=False)
        aggregated_batch.drop_column_alias()

        arg_map = {arg.key: arg.value for arg in self.node.metadata}
        start_time = int(time.time())
        auto_train_results = auto_train(
            dataset=aggregated_batch.frames,
            target=arg_map["predict"],
            tune_for_memory=arg_map.get("tune_for_memory", False),
            time_limit_s=arg_map.get("time_limit", DEFAULT_TRAIN_TIME_LIMIT),
            output_directory=self.db.catalog().get_configuration_catalog_value(
                "tmp_dir"
            ),
        )
        train_time = int(time.time()) - start_time
        model_path = os.path.join(
            self.db.catalog().get_configuration_catalog_value("model_dir"),
            self.node.name,
        )
        auto_train_results.best_model.save(model_path)
        best_score = auto_train_results.experiment_analysis.best_result["metric_score"]
        self.node.metadata.append(
            FunctionMetadataCatalogEntry("model_path", model_path)
        )

        impl_path = Path(f"{self.function_dir}/ludwig.py").absolute().as_posix()
        io_list = self._resolve_function_io(None)
        return (
            self.node.name,
            impl_path,
            self.node.function_type,
            io_list,
            self.node.metadata,
            best_score,
            train_time,
        )

    def handle_sklearn_function(self):
        """Handle sklearn functions

        Use Sklearn's regression to train models.
        """
        try_to_import_sklearn()
        from sklearn.linear_model import LinearRegression

        assert (
            len(self.children) == 1
        ), "Create sklearn function expects 1 child, finds {}.".format(
            len(self.children)
        )

        aggregated_batch_list = []
        child = self.children[0]
        for batch in child.exec():
            aggregated_batch_list.append(batch)
        aggregated_batch = Batch.concat(aggregated_batch_list, copy=False)
        aggregated_batch.drop_column_alias()

        arg_map = {arg.key: arg.value for arg in self.node.metadata}
        model = LinearRegression()
        Y = aggregated_batch.frames[arg_map["predict"]]
        aggregated_batch.frames.drop([arg_map["predict"]], axis=1, inplace=True)
        start_time = int(time.time())
        model.fit(X=aggregated_batch.frames, y=Y)
        train_time = int(time.time()) - start_time
        score = model.score(X=aggregated_batch.frames, y=Y)
        model_path = os.path.join(
            self.db.catalog().get_configuration_catalog_value("model_dir"),
            self.node.name,
        )
        pickle.dump(model, open(model_path, "wb"))
        self.node.metadata.append(
            FunctionMetadataCatalogEntry("model_path", model_path)
        )
        # Pass the prediction column name to sklearn.py
        self.node.metadata.append(
            FunctionMetadataCatalogEntry("predict_col", arg_map["predict"])
        )

        impl_path = Path(f"{self.function_dir}/sklearn.py").absolute().as_posix()
        io_list = self._resolve_function_io(None)
        return (
            self.node.name,
            impl_path,
            self.node.function_type,
            io_list,
            self.node.metadata,
            score,
            train_time,
        )

    def convert_to_numeric(self, x):
        x = re.sub("[^0-9.,]", "", str(x))
        locale.setlocale(locale.LC_ALL, "")
        x = float(locale.atof(x))
        if x.is_integer():
            return int(x)
        else:
            return x

    def handle_xgboost_function(self):
        """Handle xgboost functions

        We use the Flaml AutoML model for training xgboost models.
        """
        try_to_import_xgboost()

        assert (
            len(self.children) == 1
        ), "Create sklearn function expects 1 child, finds {}.".format(
            len(self.children)
        )

        aggregated_batch_list = []
        child = self.children[0]
        for batch in child.exec():
            aggregated_batch_list.append(batch)
        aggregated_batch = Batch.concat(aggregated_batch_list, copy=False)
        aggregated_batch.drop_column_alias()

        arg_map = {arg.key: arg.value for arg in self.node.metadata}
        from flaml import AutoML

        model = AutoML()
        settings = {
            "time_budget": arg_map.get("time_limit", DEFAULT_TRAIN_TIME_LIMIT),
            "metric": arg_map.get("metric", DEFAULT_TRAIN_REGRESSION_METRIC),
            "estimator_list": ["xgboost"],
            "task": arg_map.get("task", DEFAULT_XGBOOST_TASK),
        }
        start_time = int(time.time())
        model.fit(
            dataframe=aggregated_batch.frames, label=arg_map["predict"], **settings
        )
        train_time = int(time.time()) - start_time
        model_path = os.path.join(
            self.db.catalog().get_configuration_catalog_value("model_dir"),
            self.node.name,
        )
        pickle.dump(model, open(model_path, "wb"))
        self.node.metadata.append(
            FunctionMetadataCatalogEntry("model_path", model_path)
        )
        # Pass the prediction column to xgboost.py.
        self.node.metadata.append(
            FunctionMetadataCatalogEntry("predict_col", arg_map["predict"])
        )

        impl_path = Path(f"{self.function_dir}/xgboost.py").absolute().as_posix()
        io_list = self._resolve_function_io(None)
        best_score = model.best_loss
        return (
            self.node.name,
            impl_path,
            self.node.function_type,
            io_list,
            self.node.metadata,
            best_score,
            train_time,
        )

    def handle_ultralytics_function(self):
        """Handle Ultralytics functions"""
        try_to_import_ultralytics()

        impl_path = (
            Path(f"{self.function_dir}/yolo_object_detector.py").absolute().as_posix()
        )
        function = self._try_initializing_function(
            impl_path, function_args=get_metadata_properties(self.node)
        )
        io_list = self._resolve_function_io(function)
        return (
            self.node.name,
            impl_path,
            self.node.function_type,
            io_list,
            self.node.metadata,
        )

    def handle_forecasting_function(self):
        """Handle forecasting functions"""
        aggregated_batch_list = []
        child = self.children[0]
        for batch in child.exec():
            aggregated_batch_list.append(batch)
        aggregated_batch = Batch.concat(aggregated_batch_list, copy=False)
        aggregated_batch.drop_column_alias()

        arg_map = {arg.key: arg.value for arg in self.node.metadata}
        if not self.node.impl_path:
            impl_path = Path(f"{self.function_dir}/forecast.py").absolute().as_posix()
        else:
            impl_path = self.node.impl_path.absolute().as_posix()
        library = "statsforecast"
        supported_libraries = ["statsforecast", "neuralforecast"]

        if "horizon" not in arg_map.keys():
            raise ValueError(
                "Horizon must be provided while creating function of type FORECASTING"
            )
        try:
            horizon = int(arg_map["horizon"])
        except Exception as e:
            err_msg = f"{str(e)}. HORIZON must be integral."
            logger.error(err_msg)
            raise FunctionIODefinitionError(err_msg)

        if "library" in arg_map.keys():
            try:
                assert arg_map["library"].lower() in supported_libraries
            except Exception:
                err_msg = (
                    "EvaDB currently supports " + str(supported_libraries) + " only."
                )
                logger.error(err_msg)
                raise FunctionIODefinitionError(err_msg)

            library = arg_map["library"].lower()

        """
        The following rename is needed for statsforecast/neuralforecast, which requires the column name to be the following:
        - The unique_id (string, int or category) represents an identifier for the series.
        - The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp.
        - The y (numeric) represents the measurement we wish to forecast.
        For reference: https://nixtla.github.io/statsforecast/docs/getting-started/getting_started_short.html
        """
        aggregated_batch.rename(columns={arg_map["predict"]: "y"})
        if "time" in arg_map.keys():
            aggregated_batch.rename(columns={arg_map["time"]: "ds"})
        if "id" in arg_map.keys():
            aggregated_batch.rename(columns={arg_map["id"]: "unique_id"})

        data = aggregated_batch.frames
        if "unique_id" not in list(data.columns):
            data["unique_id"] = [1 for x in range(len(data))]

        if "ds" not in list(data.columns):
            data["ds"] = [x + 1 for x in range(len(data))]

        """
            Set or infer data frequency
        """

        if "frequency" not in arg_map.keys() or arg_map["frequency"] == "auto":
            arg_map["frequency"] = pd.infer_freq(data["ds"])
        frequency = arg_map["frequency"]
        if frequency is None:
            raise RuntimeError(
                f"Can not infer the frequency for {self.node.name}. Please explicitly set it."
            )

        season_dict = {  # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
            "H": 24,
            "M": 12,
            "Q": 4,
            "SM": 24,
            "BM": 12,
            "BMS": 12,
            "BQ": 4,
            "BH": 24,
        }

        new_freq = (
            frequency.split("-")[0] if "-" in frequency else frequency
        )  # shortens longer frequencies like Q-DEC
        season_length = season_dict[new_freq] if new_freq in season_dict else 1

        """
            Neuralforecast implementation
        """
        if library == "neuralforecast":
            try_to_import_neuralforecast()
            from neuralforecast import NeuralForecast
            from neuralforecast.auto import AutoNBEATS, AutoNHITS
            from neuralforecast.models import NBEATS, NHITS

            model_dict = {
                "AutoNBEATS": AutoNBEATS,
                "AutoNHITS": AutoNHITS,
                "NBEATS": NBEATS,
                "NHITS": NHITS,
            }

            if "model" not in arg_map.keys():
                arg_map["model"] = "NBEATS"

            if "auto" not in arg_map.keys() or (
                arg_map["auto"].lower()[0] == "t"
                and "auto" not in arg_map["model"].lower()
            ):
                arg_map["model"] = "Auto" + arg_map["model"]

            try:
                model_here = model_dict[arg_map["model"]]
            except Exception:
                err_msg = "Supported models: " + str(model_dict.keys())
                logger.error(err_msg)
                raise FunctionIODefinitionError(err_msg)
            model_args = {}

            if "auto" not in arg_map["model"].lower():
                model_args["input_size"] = 2 * horizon
                model_args["early_stop_patience_steps"] = 20
            else:
                model_args_config = {
                    "input_size": 2 * horizon,
                    "early_stop_patience_steps": 20,
                }

            if len(data.columns) >= 4:
                exogenous_columns = [
                    x for x in list(data.columns) if x not in ["ds", "y", "unique_id"]
                ]
                if "auto" not in arg_map["model"].lower():
                    model_args["hist_exog_list"] = exogenous_columns
                else:
                    model_args_config["hist_exog_list"] = exogenous_columns

                    def get_optuna_config(trial):
                        return model_args_config

                    model_args["config"] = get_optuna_config
                    model_args["backend"] = "optuna"

            model_args["h"] = horizon

            model = NeuralForecast(
                [model_here(**model_args)],
                freq=new_freq,
            )

        # """
        #     Statsforecast implementation
        # """
        else:
            if "auto" in arg_map.keys() and arg_map["auto"].lower()[0] != "t":
                raise RuntimeError(
                    "Statsforecast implementation only supports automatic hyperparameter optimization. Please set AUTO to true."
                )
            try_to_import_statsforecast()
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoTheta

            model_dict = {
                "AutoARIMA": AutoARIMA,
                "AutoCES": AutoCES,
                "AutoETS": AutoETS,
                "AutoTheta": AutoTheta,
            }

            if "model" not in arg_map.keys():
                arg_map["model"] = "ARIMA"

            if "auto" not in arg_map["model"].lower():
                arg_map["model"] = "Auto" + arg_map["model"]

            try:
                model_here = model_dict[arg_map["model"]]
            except Exception:
                err_msg = "Supported models: " + str(model_dict.keys())
                logger.error(err_msg)
                raise FunctionIODefinitionError(err_msg)

            model = StatsForecast(
                [model_here(season_length=season_length)], freq=new_freq
            )

        data["ds"] = pd.to_datetime(data["ds"])

        model_save_dir_name = library + "_" + arg_map["model"] + "_" + new_freq
        if len(data.columns) >= 4 and library == "neuralforecast":
            model_save_dir_name += "_exogenous_" + str(sorted(exogenous_columns))

        model_dir = os.path.join(
            self.db.catalog().get_configuration_catalog_value("model_dir"),
            "tsforecasting",
            model_save_dir_name,
            str(hashlib.sha256(data.to_string().encode()).hexdigest()),
        )
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        model_save_name = "horizon" + str(horizon) + ".pkl"

        model_path = os.path.join(model_dir, model_save_name)

        existing_model_files = sorted(
            os.listdir(model_dir),
            key=lambda x: int(x.split("horizon")[1].split(".pkl")[0]),
        )
        existing_model_files = [
            x
            for x in existing_model_files
            if int(x.split("horizon")[1].split(".pkl")[0]) >= horizon
        ]
        if len(existing_model_files) == 0:
            logger.info("Training, please wait...")
            for column in data.columns:
                if column != "ds" and column != "unique_id":
                    data[column] = data.apply(
                        lambda x: self.convert_to_numeric(x[column]), axis=1
                    )
            if library == "neuralforecast":
                cuda_devices_here = "0"
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    cuda_devices_here = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]

                with set_env(CUDA_VISIBLE_DEVICES=cuda_devices_here):
                    model.fit(df=data, val_size=horizon)
                    model.save(model_path, overwrite=True)
            else:
                # The following lines of code helps eliminate the math error encountered in statsforecast when only one datapoint is available in a time series
                for col in data["unique_id"].unique():
                    if len(data[data["unique_id"] == col]) == 1:
                        data = data._append(
                            [data[data["unique_id"] == col]], ignore_index=True
                        )

                model.fit(df=data[["ds", "y", "unique_id"]])
                f = open(model_path, "wb")
                pickle.dump(model, f)
                f.close()
        elif not Path(model_path).exists():
            model_path = os.path.join(model_dir, existing_model_files[-1])

        io_list = self._resolve_function_io(None)

        metadata_here = [
            FunctionMetadataCatalogEntry("model_name", arg_map["model"]),
            FunctionMetadataCatalogEntry("model_path", model_path),
            FunctionMetadataCatalogEntry(
                "predict_column_rename", arg_map.get("predict", "y")
            ),
            FunctionMetadataCatalogEntry(
                "time_column_rename", arg_map.get("time", "ds")
            ),
            FunctionMetadataCatalogEntry(
                "id_column_rename", arg_map.get("id", "unique_id")
            ),
            FunctionMetadataCatalogEntry("horizon", horizon),
            FunctionMetadataCatalogEntry("library", library),
        ]

        return (
            self.node.name,
            impl_path,
            self.node.function_type,
            io_list,
            metadata_here,
        )

    def handle_generic_function(self):
        """Handle generic functions

        Generic functions are loaded from a file. We check for inputs passed by the user during CREATE or try to load io from decorators.
        """
        impl_path = self.node.impl_path.absolute().as_posix()
        function = self._try_initializing_function(impl_path)
        io_list = self._resolve_function_io(function)

        return (
            self.node.name,
            impl_path,
            self.node.function_type,
            io_list,
            self.node.metadata,
        )

    def exec(self, *args, **kwargs):
        """Create function executor

        Calls the catalog to insert a function catalog entry.
        """
        assert (
            self.node.if_not_exists and self.node.or_replace
        ) is False, (
            "OR REPLACE and IF NOT EXISTS can not be both set for CREATE FUNCTION."
        )

        overwrite = False
        best_score = False
        train_time = False
        # check catalog if it already has this function entry
        if self.catalog().get_function_catalog_entry_by_name(self.node.name):
            if self.node.if_not_exists:
                msg = f"Function {self.node.name} already exists, nothing added."
                yield Batch(pd.DataFrame([msg]))
                return
            elif self.node.or_replace:
                # We use DropObjectExecutor to avoid bookkeeping the code. The drop function should be moved to catalog.
                from evadb.executor.drop_object_executor import DropObjectExecutor

                drop_executor = DropObjectExecutor(self.db, None)
                try:
                    drop_executor._handle_drop_function(self.node.name, if_exists=False)
                except RuntimeError:
                    pass
                else:
                    overwrite = True
            else:
                msg = f"Function {self.node.name} already exists."
                logger.error(msg)
                raise RuntimeError(msg)

        # if it's a type of HuggingFaceModel, override the impl_path
        if string_comparison_case_insensitive(self.node.function_type, "HuggingFace"):
            (
                name,
                impl_path,
                function_type,
                io_list,
                metadata,
            ) = self.handle_huggingface_function()
        elif string_comparison_case_insensitive(self.node.function_type, "ultralytics"):
            (
                name,
                impl_path,
                function_type,
                io_list,
                metadata,
            ) = self.handle_ultralytics_function()
        elif string_comparison_case_insensitive(self.node.function_type, "Ludwig"):
            (
                name,
                impl_path,
                function_type,
                io_list,
                metadata,
                best_score,
                train_time,
            ) = self.handle_ludwig_function()
        elif string_comparison_case_insensitive(self.node.function_type, "Sklearn"):
            (
                name,
                impl_path,
                function_type,
                io_list,
                metadata,
                best_score,
                train_time,
            ) = self.handle_sklearn_function()
        elif string_comparison_case_insensitive(self.node.function_type, "XGBoost"):
            (
                name,
                impl_path,
                function_type,
                io_list,
                metadata,
                best_score,
                train_time,
            ) = self.handle_xgboost_function()
        elif string_comparison_case_insensitive(self.node.function_type, "Forecasting"):
            (
                name,
                impl_path,
                function_type,
                io_list,
                metadata,
            ) = self.handle_forecasting_function()
        else:
            (
                name,
                impl_path,
                function_type,
                io_list,
                metadata,
            ) = self.handle_generic_function()

        self.catalog().insert_function_catalog_entry(
            name, impl_path, function_type, io_list, metadata
        )

        if overwrite:
            msg = f"Function {self.node.name} overwritten."
        else:
            msg = f"Function {self.node.name} added to the database."
        if best_score and train_time:
            yield Batch(
                pd.DataFrame(
                    [
                        msg,
                        "Validation Score: " + str(best_score),
                        "Training time: " + str(train_time) + " secs.",
                    ]
                )
            )
        else:
            yield Batch(pd.DataFrame([msg]))

    def _try_initializing_function(
        self, impl_path: str, function_args: Dict = {}
    ) -> FunctionCatalogEntry:
        """Attempts to initialize function given the implementation file path and arguments.

        Args:
            impl_path (str): The file path of the function implementation file.
            function_args (Dict, optional): Dictionary of arguments to pass to the function. Defaults to {}.

        Returns:
            FunctionCatalogEntry: A FunctionCatalogEntry object that represents the initialized function.

        Raises:
            RuntimeError: If an error occurs while initializing the function.
        """

        # load the function class from the file
        try:
            # loading the function class from the file
            function = load_function_class_from_file(impl_path, self.node.name)
            # initializing the function class calls the setup method internally
            function(**function_args)
        except Exception as e:
            err_msg = f"Error creating function {self.node.name}: {str(e)}"
            # logger.error(err_msg)
            raise RuntimeError(err_msg)

        return function

    def _resolve_function_io(
        self, function: FunctionCatalogEntry
    ) -> List[FunctionIOCatalogEntry]:
        """Private method that resolves the input/output definitions for a given function.
        It first searches for the input/outputs in the CREATE statement. If not found, it resolves them using decorators. If not found there as well, it raises an error.

        Args:
            function (FunctionCatalogEntry): The function for which to resolve input and output definitions.

        Returns:
            A List of FunctionIOCatalogEntry objects that represent the resolved input and
            output definitions for the function.

        Raises:
            RuntimeError: If an error occurs while resolving the function input/output
            definitions.
        """
        io_list = []
        try:
            if self.node.inputs:
                io_list.extend(self.node.inputs)
            else:
                # try to load the inputs from decorators, the inputs from CREATE statement take precedence
                io_list.extend(
                    load_io_from_function_decorators(function, is_input=True)
                )

            if self.node.outputs:
                io_list.extend(self.node.outputs)
            else:
                # try to load the outputs from decorators, the outputs from CREATE statement take precedence
                io_list.extend(
                    load_io_from_function_decorators(function, is_input=False)
                )

        except FunctionIODefinitionError as e:
            err_msg = (
                f"Error creating function, input/output definition incorrect: {str(e)}"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        return io_list
