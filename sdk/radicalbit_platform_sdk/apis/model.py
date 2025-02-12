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
        """Delete the actual `Model` from the platform."""
        invoke(
            method='DELETE',
            url=f'{self._base_url}/api/models/{self._uuid}',
            valid_response_code=200,
            func=lambda _: None,
        )

    def update_features(self, new_features: List[ColumnDefinition]) -> None:
        """Update the features of the model."""
        invoke(
            method='POST',
            url=f'{self._base_url}/api/models/{self._uuid}',
            valid_response_code=200,
            data=ModelFeatures(features=new_features).model_dump_json(),
        )
        self._features = new_features

    def get_reference_datasets(self) -> List[ModelReferenceDataset]:
        """Retrieve all reference datasets associated with the model."""
        def callback(response: requests.Response) -> List[ModelReferenceDataset]:
            try:
                adapter = TypeAdapter(List[ReferenceFileUpload])
                references = adapter.validate_python(response.json())
                return [
                    ModelReferenceDataset(
                        base_url=self._base_url,
                        model_uuid=self._uuid,
                        model_type=self._model_type,
                        reference_file_upload=ref
                    )
                    for ref in references
                ]
            except ValidationError as e:
                raise ClientError(f'Unable to parse response: {response.text}') from e

        return invoke(
            method='GET',
            url=f'{self._base_url}/api/models/{self._uuid}/reference/all',
            valid_response_code=200,
            func=callback,
        )

    def get_current_datasets(self) -> List[ModelCurrentDataset]:
        """Retrieve all current datasets associated with the model."""
        def callback(response: requests.Response) -> List[ModelCurrentDataset]:
            try:
                adapter = TypeAdapter(List[CurrentFileUpload])
                references = adapter.validate_python(response.json())
                return [
                    ModelCurrentDataset(
                        base_url=self._base_url,
                        model_uuid=self._uuid,
                        model_type=self._model_type,
                        current_file_upload=ref
                    )
                    for ref in references
                ]
            except ValidationError as e:
                raise ClientError(f'Unable to parse response: {response.text}') from e

        return invoke(
            method='GET',
            url=f'{self._base_url}/api/models/{self._uuid}/current/all',
            valid_response_code=200,
            func=callback,
        )

    def load_reference_dataset(
        self,
        file_path: str,
        bucket: str,
        object_name: Optional[str] = None,
        aws_credentials: Optional[AwsCredentials] = None,
        separator: str = ',',
    ) -> ModelReferenceDataset:
        """Upload reference dataset to S3 and bind it to the model."""
        self._validate_file_headers(file_path, separator)
        object_name = object_name or f'{self._uuid}/reference/{os.path.basename(file_path)}'
        self._upload_file_to_s3(file_path, bucket, object_name, aws_credentials)
        return self._bind_reference_dataset(f's3://{bucket}/{object_name}', separator)

    def bind_reference_dataset(
        self,
        dataset_url: str,
        aws_credentials: Optional[AwsCredentials] = None,
        separator: str = ',',
    ) -> ModelReferenceDataset:
        """Bind an existing reference dataset file already uploaded to S3 to the model."""
        self._validate_file_headers_from_url(dataset_url, aws_credentials, separator)
        return self._bind_reference_dataset(dataset_url, separator)

    def load_current_dataset(
        self,
        file_path: str,
        bucket: str,
        correlation_id_column: Optional[str] = None,
        object_name: Optional[str] = None,
        aws_credentials: Optional[AwsCredentials] = None,
        separator: str = ',',
    ) -> ModelCurrentDataset:
        """Upload current dataset to S3 and bind it to the model."""
        required_headers = self._required_headers()
        if correlation_id_column:
            required_headers.append(correlation_id_column)
        required_headers.append(self._timestamp.name)
        self._validate_file_headers(file_path, separator, required_headers)
        object_name = object_name or f'{self._uuid}/current/{os.path.basename(file_path)}'
        self._upload_file_to_s3(file_path, bucket, object_name, aws_credentials)
        return self._bind_current_dataset(f's3://{bucket}/{object_name}', separator, correlation_id_column)

    def bind_current_dataset(
        self,
        dataset_url: str,
        correlation_id_column: str,
        aws_credentials: Optional[AwsCredentials] = None,
        separator: str = ',',
    ) -> ModelCurrentDataset:
        """Bind an existing current dataset file already uploaded to S3 to the model."""
        required_headers = self._required_headers()
        required_headers.append(correlation_id_column)
        required_headers.append(self._timestamp.name)
        self._validate_file_headers_from_url(dataset_url, aws_credentials, separator, required_headers)
        return self._bind_current_dataset(dataset_url, separator, correlation_id_column)

    def _validate_file_headers(self, file_path: str, separator: str, required_headers: Optional[List[str]] = None) -> None:
        file_headers = pd.read_csv(file_path, nrows=0, delimiter=separator).columns.tolist()
        required_headers = required_headers or self._required_headers()
        if not set(required_headers).issubset(file_headers):
            raise ClientError(f'File {file_path} does not contain all required columns: {required_headers}')

    def _validate_file_headers_from_url(self, dataset_url: str, aws_credentials: Optional[AwsCredentials], separator: str, required_headers: Optional[List[str]] = None) -> None:
        required_headers = required_headers or self._required_headers()
        s3_client = self._get_s3_client(aws_credentials)
        url_parts = dataset_url.replace('s3://', '').split('/')
        response = s3_client.get_object(Bucket=url_parts[0], Key='/'.join(url_parts[1:]))
        file_headers = response['Body'].readline().decode('UTF-8').strip().split(separator)
        if not set(required_headers).issubset(file_headers):
            raise ClientError(f'File {dataset_url} does not contain all required columns: {required_headers}')

    def _upload_file_to_s3(self, file_path: str, bucket: str, object_name: str, aws_credentials: Optional[AwsCredentials]) -> None:
        try:
            s3_client = self._get_s3_client(aws_credentials)
            s3_client.upload_file(
                file_path,
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
            raise ClientError(f'Unable to upload file {file_path} to remote storage: {e}') from e

    def _bind_reference_dataset(self, dataset_url: str, separator: str) -> ModelReferenceDataset:
        file_ref = FileReference(file_url=dataset_url, separator=separator)
        response = invoke(
            method='POST',
            url=f'{self._base_url}/api/models/{self._uuid}/reference/bind',
            valid_response_code=200,
            data=file_ref.model_dump_json(),
        )
        return self._create_model_reference_dataset(response)

    def _bind_current_dataset(self, dataset_url: str, separator: str, correlation_id_column: Optional[str]) -> ModelCurrentDataset:
        file_ref = FileReference(file_url=dataset_url, separator=separator, correlation_id_column=correlation_id_column)
        response = invoke(
            method='POST',
            url=f'{self._base_url}/api/models/{self._uuid}/current/bind',
            valid_response_code=200,
            data=file_ref.model_dump_json(),
        )
        return self._create_model_current_dataset(response)

    def _get_s3_client(self, aws_credentials: Optional[AwsCredentials]) -> boto3.client:
        return boto3.client(
            's3',
            aws_access_key_id=aws_credentials.access_key_id if aws_credentials else None,
            aws_secret_access_key=aws_credentials.secret_access_key if aws_credentials else None,
            region_name=aws_credentials.default_region if aws_credentials else None,
            endpoint_url=aws_credentials.endpoint_url if aws_credentials else None,
        )

    def _required_headers(self) -> List[str]:
        model_columns = self._features + self._outputs.output
        model_columns.append(self._target)
        return [model_column.name for model_column in model_columns]

    def _create_model_reference_dataset(self, response: requests.Response) -> ModelReferenceDataset:
        try:
            reference_file_upload = ReferenceFileUpload.model_validate(response.json())
            return ModelReferenceDataset(
                base_url=self._base_url,
                model_uuid=self._uuid,
                model_type=self._model_type,
                reference_file_upload=reference_file_upload
            )
        except ValidationError as e:
            raise ClientError(f'Unable to parse response: {response.text}') from e

    def _create_model_current_dataset(self, response: requests.Response) -> ModelCurrentDataset:
        try:
            current_file_upload = CurrentFileUpload.model_validate(response.json())
            return ModelCurrentDataset(
                base_url=self._base_url,
                model_uuid=self._uuid,
                model_type=self._model_type,
                current_file_upload=current_file_upload
            )
        except ValidationError as e:
            raise ClientError(f'Unable to parse response: {response.text}') from e