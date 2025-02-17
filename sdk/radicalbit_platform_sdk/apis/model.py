import os
from typing import List, Optional
from uuid import UUID

import boto3
from botocore.exceptions import ClientError as BotoClientError
import pandas as pd
from pydantic import TypeAdapter, ValidationError
import requests

from radicalbit_platform_sdk.apis import ModelCurrentDataset, ModelReferenceDataset
from radicalbit_platform_sdk.commons import invoke
from radicalbit_platform_sdk.errors import ClientError
from radicalbit_platform_sdk.models import (
    AwsCredentials,
    ColumnDefinition,
    CurrentFileUpload,
    DataType,
    FileReference,
    Granularity,
    ModelDefinition,
    ModelType,
    OutputType,
    ReferenceFileUpload,
)


class Model:
    def __init__(self, base_url: str, definition: ModelDefinition) -> None:
        self._base_url = base_url
        self._uuid = definition.uuid
        self._name = definition.name
        self._description = definition.description
        self._model_type = definition.model_type
        self._data_type = definition.data_type
        self._granularity = definition.granularity
        self._features = definition.features
        self._target = definition.target
        self._timestamp = definition.timestamp
        self._outputs = definition.outputs
        self._frameworks = definition.frameworks
        self._algorithm = definition.algorithm

    def uuid(self) -> UUID:
        return self._uuid

    def name(self) -> str:
        return self._name

    def description(self) -> Optional[str]:
        return self._description

    def model_type(self) -> ModelType:
        return self._model_type

    def data_type(self) -> DataType:
        return self._data_type

    def granularity(self) -> Granularity:
        return self._granularity

    def features(self) -> List[ColumnDefinition]:
        return self._features

    def target(self) -> ColumnDefinition:
        return self._target

    def timestamp(self) -> ColumnDefinition:
        return self._timestamp

    def outputs(self) -> OutputType:
        return self._outputs

    def frameworks(self) -> Optional[str]:
        return self._frameworks

    def algorithm(self) -> Optional[str]:
        return self._algorithm

    def delete(self) -> None:
        """Delete the actual `Model` from the platform"""
        invoke(
            method='DELETE',
            url=f'{self._base_url}/api/models/{str(self._uuid)}',
            valid_response_code=200,
            func=lambda _: None,
        )

    def get_reference_datasets(self) -> List[ModelReferenceDataset]:
        def callback(response: requests.Response) -> List[ModelReferenceDataset]:
            try:
                adapter = TypeAdapter(List[ReferenceFileUpload])
                references = adapter.validate_python(response.json())
                return [
                    ModelReferenceDataset(self._base_url, self._uuid, self._model_type, ref)
                    for ref in references
                ]
            except ValidationError as e:
                raise ClientError(f'Unable to parse response: {response.text}') from e

        return invoke(
            method='GET',
            url=f'{self._base_url}/api/models/{str(self._uuid)}/reference/all',
            valid_response_code=200,
            func=callback,
        )

    def get_current_datasets(self) -> List[ModelCurrentDataset]:
        def callback(response: requests.Response) -> List[ModelCurrentDataset]:
            try:
                adapter = TypeAdapter(List[CurrentFileUpload])
                references = adapter.validate_python(response.json())
                return [
                    ModelCurrentDataset(self._base_url, self._uuid, self._model_type, ref)
                    for ref in references
                ]
            except ValidationError as e:
                raise ClientError(f'Unable to parse response: {response.text}') from e

        return invoke(
            method='GET',
            url=f'{self._base_url}/api/models/{str(self._uuid)}/current/all',
            valid_response_code=200,
            func=callback,
        )

    def load_reference_dataset(
        self,
        file_name: str,
        bucket: str,
        object_name: Optional[str] = None,
        aws_credentials: Optional[AwsCredentials] = None,
        separator: str = ',',
    ) -> ModelReferenceDataset:
        """Upload reference dataset to an S3 bucket and then bind it inside the platform.

        Raises `ClientError` in case S3 upload fails.

        :param file_name: The name of the reference file.
        :param bucket: The name of the S3 bucket.
        :param object_name: The optional name of the object uploaded to S3. Default value is None.
        :param aws_credentials: AWS credentials used to connect to S3 bucket. Default value is None.
        :param separator: Optional value to define separator used inside CSV file. Default value is ","
        :return: An instance of `ModelReferenceDataset` representing the reference dataset
        """

        file_headers = pd.read_csv(file_name, nrows=0, delimiter=separator).columns.tolist()
        required_headers = self._required_headers()

        if set(required_headers).issubset(file_headers):
            if object_name is None:
                object_name = f'{self._uuid}/reference/{os.path.basename(file_name)}'

            s3_client = self._get_s3_client(aws_credentials)
            try:
                s3_client.upload_file(
                    file_name,
                    bucket,
                    object_name,
                    ExtraArgs={
                        'Metadata': {
                            'model_uuid': str(self._uuid),
                            'model_name': self._name,
                            'file_type': 'reference',
                        }
                    },
                )
            except BotoClientError as e:
                raise ClientError(f'Unable to upload file {file_name} to remote storage: {e}') from e

            return self._bind_reference_dataset(f's3://{bucket}/{object_name}', separator)

        raise ClientError(f'File {file_name} does not contain all defined columns: {required_headers}')

    def bind_reference_dataset(
        self,
        dataset_url: str,
        aws_credentials: Optional[AwsCredentials] = None,
        separator: str = ',',
    ) -> ModelReferenceDataset:
        """Bind an existing reference dataset file already uploaded to S3 to a `Model`

        :param dataset_url: The URL of the file already uploaded inside S3
        :param aws_credentials: AWS credentials used to connect to S3 bucket. Default value is None.
        :param separator: Optional value to define separator used inside CSV file. Default value is ","
        :return: An instance of `ModelReferenceDataset` representing the reference dataset
        """

        s3_client = self._get_s3_client(aws_credentials)
        try:
            file_headers = self._get_file_headers(s3_client, dataset_url, separator)
            required_headers = self._required_headers()

            if set(required_headers).issubset(file_headers):
                return self._bind_reference_dataset(dataset_url, separator)

            raise ClientError(f'File {dataset_url} does not contain all defined columns: {required_headers}')
        except BotoClientError as e:
            raise ClientError(f'Unable to get file {dataset_url} from remote storage: {e}') from e

    def load_current_dataset(
        self,
        file_name: str,
        bucket: str,
        correlation_id_column: Optional[str] = None,
        object_name: Optional[str] = None,
        aws_credentials: Optional[AwsCredentials] = None,
        separator: str = ',',
    ) -> ModelCurrentDataset:
        """Upload current dataset to an S3 bucket and then bind it inside the platform.

        Raises `ClientError` in case S3 upload fails.

        :param file_name: The name of the reference file.
        :param bucket: The name of the S3 bucket.
        :param correlation_id_column: The name of the column used for correlation id
        :param object_name: The optional name of the object uploaded to S3. Default value is None.
        :param aws_credentials: AWS credentials used to connect to S3 bucket. Default value is None.
        :param separator: Optional value to define separator used inside CSV file. Default value is ","
        :return: An instance of `ModelCurrentDataset` representing the current dataset
        """

        file_headers = pd.read_csv(file_name, nrows=0, delimiter=separator).columns.tolist()
        required_headers = self._required_headers()
        if correlation_id_column:
            required_headers.append(correlation_id_column)
        required_headers.append(self._timestamp.name)

        if set(required_headers).issubset(file_headers):
            if object_name is None:
                object_name = f'{self._uuid}/current/{os.path.basename(file_name)}'

            s3_client = self._get_s3_client(aws_credentials)
            try:
                s3_client.upload_file(
                    file_name,
                    bucket,
                    object_name,
                    ExtraArgs={
                        'Metadata': {
                            'model_uuid': str(self._uuid),
                            'model_name': self._name,
                            'file_type': 'current',
                        }
                    },
                )
            except BotoClientError as e:
                raise ClientError(f'Unable to upload file {file_name} to remote storage: {e}') from e

            return self._bind_current_dataset(f's3://{bucket}/{object_name}', separator, correlation_id_column)

        raise ClientError(f'File {file_name} does not contain all defined columns: {required_headers}')

    def bind_current_dataset(
        self,
        dataset_url: str,
        correlation_id_column: str,
        aws_credentials: Optional[AwsCredentials] = None,
        separator: str = ',',
    ) -> ModelCurrentDataset:
        """Bind an existing current dataset file already uploaded to S3 to a `Model`

        :param dataset_url: The URL of the file already uploaded inside S3
        :param correlation_id_column: The name of the column used for correlation id
        :param aws_credentials: AWS credentials used to connect to S3 bucket. Default value is None.
        :param separator: Optional value to define separator used inside CSV file. Default value is ","
        :return: An instance of `ModelCurrentDataset` representing the current dataset
        """

        s3_client = self._get_s3_client(aws_credentials)
        try:
            file_headers = self._get_file_headers(s3_client, dataset_url, separator)
            required_headers = self._required_headers()
            required_headers.append(correlation_id_column)
            required_headers.append(self._timestamp.name)

            if set(required_headers).issubset(file_headers):
                return self._bind_current_dataset(dataset_url, separator, correlation_id_column)

            raise ClientError(f'File {dataset_url} does not contain all defined columns: {required_headers}')
        except BotoClientError as e:
            raise ClientError(f'Unable to get file {dataset_url} from remote storage: {e}') from e

    def update_features(self, new_features: List[ColumnDefinition]) -> None:
        """Update the model features.

        :param new_features: A list of new features to be set for the model.
        """
        invoke(
            method='POST',
            url=f'{self._base_url}/api/models/{str(self._uuid)}',
            valid_response_code=200,
            func=lambda _: None,
            data=ModelFeatures(features=new_features).model_dump_json(),
        )
        self._features = new_features

    def _bind_reference_dataset(
        self,
        dataset_url: str,
        separator: str,
    ) -> ModelReferenceDataset:
        def callback(response: requests.Response) -> ModelReferenceDataset:
            try:
                response = ReferenceFileUpload.model_validate(response.json())
                return ModelReferenceDataset(self._base_url, self._uuid, self._model_type, response)
            except ValidationError as e:
                raise ClientError(f'Unable to parse response: {response.text}') from e

        file_ref = FileReference(file_url=dataset_url, separator=separator)
        return invoke(
            method='POST',
            url=f'{self._base_url}/api/models/{str(self._uuid)}/reference/bind',
            valid_response_code=200,
            func=callback,
            data=file_ref.model_dump_json(),
        )

    def _bind_current_dataset(
        self,
        dataset_url: str,
        separator: str,
        correlation_id_column: Optional[str] = None,
    ) -> ModelCurrentDataset:
        def callback(response: requests.Response) -> ModelCurrentDataset:
            try:
                response = CurrentFileUpload.model_validate(response.json())
                return ModelCurrentDataset(self._base_url, self._uuid, self._model_type, response)
            except ValidationError as e:
                raise ClientError(f'Unable to parse response: {response.text}') from e

        file_ref = FileReference(
            file_url=dataset_url,
            separator=separator,
            correlation_id_column=correlation_id_column,
        )
        return invoke(
            method='POST',
            url=f'{self._base_url}/api/models/{str(self._uuid)}/current/bind',
            valid_response_code=200,
            func=callback,
            data=file_ref.model_dump_json(),
        )

    def _required_headers(self) -> List[str]:
        model_columns = self._features + self._outputs.output
        model_columns.append(self._target)
        return [model_column.name for model_column in model_columns]

    def _get_s3_client(self, aws_credentials: Optional[AwsCredentials] = None) -> boto3.client:
        return boto3.client(
            's3',
            aws_access_key_id=aws_credentials.access_key_id if aws_credentials else None,
            aws_secret_access_key=aws_credentials.secret_access_key if aws_credentials else None,
            region_name=aws_credentials.default_region if aws_credentials else None,
            endpoint_url=aws_credentials.endpoint_url if aws_credentials else None,
        )

    def _get_file_headers(self, s3_client: boto3.client, dataset_url: str, separator: str) -> List[str]:
        url_parts = dataset_url.replace('s3://', '').split('/')
        chunks_iterator = s3_client.get_object(Bucket=url_parts[0], Key='/'.join(url_parts[1:]))['Body'].iter_chunks()
        chunks = ''
        for chunk in (c for c in chunks_iterator if '\n' not in chunks):
            chunks += chunk.decode('UTF-8')
        return chunks.split('\n')[0].split(separator)