import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "image-segmentation" not in ALLOWED_TASKS,
    "image-segmentation not implemented",
)

class ImageSegementationTestCase(TestCase):
    def setUp(self):
        model_id = TESTABLE_MODELS["image-segmentation"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = model_id
        os.environ["TASK"] = "image-segmentation"
        from app.main import app

        self.app = app

    @classmethod
    def setUpClass(cls):
        from app.main import get_pipeline

        get_pipeline.cache_clear()

    def tearDown(self):
        if self.old_model_id is not None:
            os.environ["MODEL_ID"] = self.old_model_id
        else:
            del os.environ["MODEL_ID"]
        if self.old_task is not None:
            os.environ["TASK"] = self.old_task
        else:
            del os.environ["TASK"]

    def read(self, filename: str) -> bytes:
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", filename)
        with open(filename, "rb") as f:
            bpayload = f.read()
        return bpayload

    def test_simple(self):
        bpayload = self.read("sample.tif")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)


        #edit correct test for pred np.array here
        """
        self.assertEqual(type(content), list)
        self.assertEqual(set(type(el) for el in content), {dict})
        self.assertEqual(
            set((k, type(v)) for el in content for (k, v) in el.items()),
            {("label", str), ("score", float)},
        )
        """

