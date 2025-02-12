from radicalbit_platform_sdk.commons import invoke
from radicalbit_platform_sdk.models import (
    JobStatus,
    ReferenceFileUpload,
    DatasetStats,
    ModelQuality,
    DataQuality,
    ModelType,
    BinaryClassificationModelQuality,
    BinaryClassificationDataQuality,
)
from radicalbit_platform_sdk.errors import ClientError
from pydantic import ValidationError
from typing import Optional
import requests
from uuid import UUID


class ModelReferenceDataset:
    def __init__(
        self,
        base_url: str,
        model_uuid: UUID,
        model_type: ModelType,
        upload: Optional[ReferenceFileUpload] = None,
    ) -> None:
        self.__base_url = base_url
        self.__model_uuid = model_uuid
        self.__model_type = model_type
        self.__uuid = upload.uuid if upload else None
        self.__path = upload.path if upload else None
        self.__date = upload.date if upload else None
        self.__status = upload.status if upload else None
        self.__statistics = None
        self.__model_metrics = None
        self.__data_metrics = None

    def uuid(self) -> Optional[UUID]:
        return self.__uuid

    def path(self) -> Optional[str]:
        return self.__path

    def date(self) -> Optional[str]:
        return self.__date

    def status(self) -> Optional[str]:
        return self.__status

    def statistics(self) -> Optional[DatasetStats]:
        """\n        Get statistics about the current dataset\n\n        :return: The `DatasetStats` if exists\n        """

        def __callback(
            response: requests.Response,
        ) -> tuple[JobStatus, Optional[DatasetStats]]:
            try:
                response_json = response.json()
                job_status = JobStatus(response_json["jobStatus"])
                if "statistics" in response_json:
                    return job_status, DatasetStats.model_validate(
                        response_json["statistics"]
                    )
                else:
                    return job_status, None
            except KeyError as _:
                raise ClientError(f"Unable to parse response: {response.text}")
            except ValidationError as _:
                raise ClientError(f"Unable to parse response: {response.text}")

        if self.__status in [JobStatus.ERROR, JobStatus.IMPORTING]:
            self.__statistics = None
        elif self.__status == JobStatus.SUCCEEDED:
            if self.__statistics is None:
                _, stats = invoke(
                    method="GET",
                    url=f"{self.__base_url}/api/models/{str(self.__model_uuid)}/reference/statistics",
                    valid_response_code=200,
                    func=__callback,
                )
                self.__statistics = stats

        return self.__statistics

    def data_quality(self) -> Optional[DataQuality]:
        """\n        Get data quality metrics about the current dataset\n\n        :return: The `DataQuality` if exists\n        """

        def __callback(response: requests.Response) -> Optional[DataQuality]:
            try:
                response_json = response.json()
                job_status = JobStatus(response_json["jobStatus"])
                if "dataQuality" in response_json:
                    if self.__model_type == ModelType.BINARY:
                        return (
                            job_status,
                            BinaryClassificationDataQuality.model_validate(
                                response_json["dataQuality"]
                            ),
                        )
                    else:
                        raise ClientError(
                            "Unable to parse get metrics for non-binary models"
                        )
                else:
                    return job_status, None
            except KeyError as _:
                raise ClientError(f"Unable to parse response: {response.text}")
            except ValidationError as _:
                raise ClientError(f"Unable to parse response: {response.text}")

        if self.__status in [JobStatus.ERROR, JobStatus.IMPORTING]:
            self.__data_metrics = None
        elif self.__status == JobStatus.SUCCEEDED:
            if self.__data_metrics is None:
                _, metrics = invoke(
                    method="GET",
                    url=f"{self.__base_url}/api/models/{str(self.__model_uuid)}/reference/data-quality",
                    valid_response_code=200,
                    func=__callback,
                )
                self.__data_metrics = metrics

        return self.__data_metrics

    def model_quality(self) -> Optional[ModelQuality]:
        """\n        Get model quality metrics about the current dataset\n\n        :return: The `ModelQuality` if exists\n        """

        def __callback(
            response: requests.Response,
        ) -> tuple[JobStatus, Optional[ModelQuality]]:
            try:
                response_json = response.json()
                job_status = JobStatus(response_json["jobStatus"])
                if "modelQuality" in response_json:
                    if self.__model_type == ModelType.BINARY:
                        return (
                            job_status,
                            BinaryClassificationModelQuality.model_validate(
                                response_json["modelQuality"]
                            ),
                        )
                    else:
                        raise ClientError(
                            "Unable to parse get metrics for non-binary models"
                        )
                else:
                    return job_status, None
            except KeyError as _:
                raise ClientError(f"Unable to parse response: {response.text}")
            except ValidationError as _:
                raise ClientError(f"Unable to parse response: {response.text}")

        if self.__status in [JobStatus.ERROR, JobStatus.IMPORTING]:
            self.__model_metrics = None
        elif self.__status == JobStatus.SUCCEEDED:
            if self.__model_metrics is None:
                _, metrics = invoke(
                    method="GET",
                    url=f"{self.__base_url}/api/models/{str(self.__model_uuid)}/reference/model-quality",
                    valid_response_code=200,
                    func=__callback,
                )
                self.__model_metrics = metrics

        return self.__model_metrics