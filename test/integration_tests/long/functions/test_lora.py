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

import unittest
from test.markers import stable_diffusion_skip_marker
from test.util import get_evadb_for_testing
from unittest.mock import patch

from evadb.server.command_handler import execute_query_fetch_all


class LoRATest(unittest.TestCase):
    def setUp(self) -> None:
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        from pathlib import Path
 
        # get the path/directory
        folder_dir = 'evadb/test/integration_tests/long/functions/test_images_for_lora'
        
        # iterate over files in
        # that directory
        images = Path(folder_dir).glob('*.png, *.jpg, *.jpeg, *.webp')
        for image in images:
            print(image)
            create_table_query = f"""LOAD IMAGE '{image}' INTO MyImages;"""
            execute_query_fetch_all(self.evadb, create_table_query)

        # test_prompts = ["pink cat riding a rocket to the moon"]

        # for prompt in test_prompts:
        #     insert_query = f"""INSERT INTO ImageGen (prompt) VALUES ('{prompt}')"""
        #     execute_query_fetch_all(self.evadb, insert_query)

    def tearDown(self) -> None:
        execute_query_fetch_all(self.evadb, "DROP TABLE IF EXISTS MyImages;")

    @patch("replicate.run", return_value=[{"response": "mocked response"}])
    def test_lora_image_training(self, mock_replicate_run):
        function_name = "StableDiffusionLoRA"

        execute_query_fetch_all(self.evadb, f"DROP FUNCTION IF EXISTS {function_name};")

        create_function_query = f"""CREATE FUNCTION IF NOT EXISTS{function_name}
            IMPL 'evadb/functions/stable_diffusion.py';
        """
        execute_query_fetch_all(self.evadb, create_function_query)

        # gpt_query = f"SELECT {function_name}(prompt) FROM ImageGen;"
        # output_batch = execute_query_fetch_all(self.evadb, gpt_query)

        self.assertEqual(output_batch, ["lora.response"])
        mock_replicate_run.assert_called_once_with(
            "cloneofsimo/lora-training:b2a308762e36ac48d16bfadc03a65493fe6e799f429f7941639a6acec5b276cc",
            input={instance_data: "images.zip", task: "style"},
        )
