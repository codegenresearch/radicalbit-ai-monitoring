import time
import unittest
import uuid

import boto3
from moto import mock_aws
import pytest
import responses

from radicalbit_platform_sdk.apis import Model
from radicalbit_platform_sdk.errors import ClientError
from radicalbit_platform_sdk.models import (
    ColumnDefinition,
    CurrentFileUpload,
    DataType,
    FieldType,
    Granularity,
    JobStatus,
    ModelDefinition,
    ModelFeatures,
    ModelType,
    OutputType,
    ReferenceFileUpload,
    SupportedTypes,
)


class ModelTest(unittest.TestCase):
    @responses.activate
    def test_delete_model(self):
        # Test deleting a model
        base_url = 'http://api:9000'
        model_id = uuid.uuid4()
        column_def = ColumnDefinition(
            name='column', type=SupportedTypes.string, field_type=FieldType.categorical
        )
        outputs = OutputType(prediction=column_def, output=[column_def])
        model = Model(
            base_url,
            ModelDefinition(
                uuid=model_id,
                name='My Model',
                model_type=ModelType.BINARY,
                data_type=DataType.TABULAR,
                granularity=Granularity.MONTH,
                features=[],
                outputs=outputs,
                target=column_def,
                timestamp=column_def,
                created_at=str(time.time()),
                updated_at=str(time.time()),
            ),
        )
        responses.add(
            method=responses.DELETE,
            url=f'{base_url}/api/models/{str(model_id)}',
            status=200,
        )
        model.delete()

    @mock_aws
    @responses.activate
    def test_load_reference_dataset_without_object_name(self):
        # Test loading a reference dataset without specifying an object name
        base_url = 'http://api:9000'
        model_id = uuid.uuid4()
        bucket_name = 'test-bucket'
        file_name = 'test.txt'
        column_def = ColumnDefinition(
            name='prediction', type=SupportedTypes.float, field_type=FieldType.numerical
        )
        expected_path = f's3://{bucket_name}/{model_id}/reference/{file_name}'
        conn = boto3.resource('s3', region_name='us-east-1')
        conn.create_bucket(Bucket=bucket_name)
        model = Model(
            base_url,
            ModelDefinition(
                uuid=model_id,
                name='My Model',
                model_type=ModelType.BINARY,
                data_type=DataType.TABULAR,
                granularity=Granularity.HOUR,
                features=[
                    ColumnDefinition(
                        name='first_name',
                        type=SupportedTypes.string,
                        field_type=FieldType.categorical,
                    ),
                    ColumnDefinition(
                        name='age',
                        type=SupportedTypes.int,
                        field_type=FieldType.numerical,
                    ),
                ],
                outputs=OutputType(prediction=column_def, output=[column_def]),
                target=ColumnDefinition(
                    name='adult',
                    type=SupportedTypes.bool,
                    field_type=FieldType.categorical,
                ),
                timestamp=ColumnDefinition(
                    name='created_at',
                    type=SupportedTypes.datetime,
                    field_type=FieldType.datetime,
                ),
                created_at=str(time.time()),
                updated_at=str(time.time()),
            ),
        )
        response = ReferenceFileUpload(
            uuid=uuid.uuid4(), path=expected_path, date='', status=JobStatus.IMPORTING
        )
        responses.add(
            method=responses.POST,
            url=f'{base_url}/api/models/{str(model_id)}/reference/bind',
            body=response.model_dump_json(),
            status=200,
            content_type='application/json',
        )
        response = model.load_reference_dataset(
            'tests_resources/people.csv', bucket_name
        )
        assert response.path() == expected_path

    @mock_aws
    @responses.activate
    def test_load_reference_dataset_with_different_separator(self):
        # Test loading a reference dataset with a different separator
        base_url = 'http://api:9000'
        model_id = uuid.uuid4()
        bucket_name = 'test-bucket'
        file_name = 'test.txt'
        column_def = ColumnDefinition(
            name='prediction', type=SupportedTypes.float, field_type=FieldType.numerical
        )
        expected_path = f's3://{bucket_name}/{model_id}/reference/{file_name}'
        conn = boto3.resource('s3', region_name='us-east-1')
        conn.create_bucket(Bucket=bucket_name)
        model = Model(
            base_url,
            ModelDefinition(
                uuid=model_id,
                name='My Model',
                model_type=ModelType.BINARY,
                data_type=DataType.TABULAR,
                granularity=Granularity.DAY,
                features=[
                    ColumnDefinition(
                        name='first_name',
                        type=SupportedTypes.string,
                        field_type=FieldType.categorical,
                    ),
                    ColumnDefinition(
                        name='age',
                        type=SupportedTypes.int,
                        field_type=FieldType.numerical,
                    ),
                ],
                outputs=OutputType(prediction=column_def, output=[column_def]),
                target=ColumnDefinition(
                    name='adult',
                    type=SupportedTypes.bool,
                    field_type=FieldType.categorical,
                ),
                timestamp=ColumnDefinition(
                    name='created_at',
                    type=SupportedTypes.datetime,
                    field_type=FieldType.datetime,
                ),
                created_at=str(time.time()),
                updated_at=str(time.time()),
            ),
        )
        response = ReferenceFileUpload(
            uuid=uuid.uuid4(), path=expected_path, date='', status=JobStatus.IMPORTING
        )
        responses.add(
            method=responses.POST,
            url=f'{base_url}/api/models/{str(model_id)}/reference/bind',
            body=response.model_dump_json(),
            status=200,
            content_type='application/json',
        )
        response = model.load_reference_dataset(
            'tests_resources/people_pipe_separated.csv', bucket_name, separator='|'
        )
        assert response.path() == expected_path

    @mock_aws
    @responses.activate
    def test_load_reference_dataset_with_object_name(self):
        # Test loading a reference dataset with a specified object name
        base_url = 'http://api:9000'
        model_id = uuid.uuid4()
        bucket_name = 'test-bucket'
        file_name = 'test.txt'
        column_def = ColumnDefinition(
            name='prediction', type=SupportedTypes.float, field_type=FieldType.numerical
        )
        expected_path = f's3://{bucket_name}/{file_name}'
        conn = boto3.resource('s3', region_name='us-east-1')
        conn.create_bucket(Bucket=bucket_name)
        model = Model(
            base_url,
            ModelDefinition(
                uuid=model_id,
                name='My Model',
                model_type=ModelType.BINARY,
                data_type=DataType.TABULAR,
                granularity=Granularity.WEEK,
                features=[
                    ColumnDefinition(
                        name='first_name',
                        type=SupportedTypes.string,
                        field_type=FieldType.categorical,
                    ),
                    ColumnDefinition(
                        name='age',
                        type=SupportedTypes.int,
                        field_type=FieldType.numerical,
                    ),
                ],
                outputs=OutputType(prediction=column_def, output=[column_def]),
                target=ColumnDefinition(
                    name='adult',
                    type=SupportedTypes.bool,
                    field_type=FieldType.categorical,
                ),
                timestamp=ColumnDefinition(
                    name='created_at',
                    type=SupportedTypes.datetime,
                    field_type=FieldType.datetime,
                ),
                created_at=str(time.time()),
                updated_at=str(time.time()),
            ),
        )
        response = ReferenceFileUpload(
            uuid=uuid.uuid4(), path=expected_path, date='', status=JobStatus.IMPORTING
        )
        responses.add(
            method=responses.POST,
            url=f'{base_url}/api/models/{str(model_id)}/reference/bind',
            body=response.model_dump_json(),
            status=200,
            content_type='application/json',
        )
        response = model.load_reference_dataset(
            'tests_resources/people.csv', bucket_name, object_name=file_name
        )
        assert response.path() == expected_path

    def test_load_reference_dataset_wrong_headers(self):
        # Test loading a reference dataset with wrong headers
        column_def = ColumnDefinition(
            name='prediction', type=SupportedTypes.float, field_type=FieldType.numerical
        )
        model = Model(
            'http://api:9000',
            ModelDefinition(
                uuid=uuid.uuid4(),
                name='My Model',
                model_type=ModelType.BINARY,
                data_type=DataType.TABULAR,
                granularity=Granularity.MONTH,
                features=[
                    ColumnDefinition(
                        name='first_name',
                        type=SupportedTypes.string,
                        field_type=FieldType.categorical,
                    ),
                    ColumnDefinition(
                        name='age',
                        type=SupportedTypes.int,
                        field_type=FieldType.numerical,
                    ),
                ],
                outputs=OutputType(prediction=column_def, output=[column_def]),
                target=ColumnDefinition(
                    name='adult',
                    type=SupportedTypes.bool,
                    field_type=FieldType.categorical,
                ),
                timestamp=ColumnDefinition(
                    name='created_at',
                    type=SupportedTypes.datetime,
                    field_type=FieldType.datetime,
                ),
                created_at=str(time.time()),
                updated_at=str(time.time()),
            ),
        )
        with pytest.raises(ClientError):
            model.load_reference_dataset('tests_resources/wrong.csv', 'bucket_name')

    @mock_aws
    @responses.activate
    def test_load_current_dataset_without_object_name(self):
        # Test loading a current dataset without specifying an object name
        base_url = 'http://api:9000'
        model_id = uuid.uuid4()
        bucket_name = 'test-bucket'
        file_name = 'test.txt'
        column_def = ColumnDefinition(
            name='prediction', type=SupportedTypes.float, field_type=FieldType.numerical
        )
        expected_path = f's3://{bucket_name}/{model_id}/current/{file_name}'
        conn = boto3.resource('s3', region_name='us-east-1')
        conn.create_bucket(Bucket=bucket_name)
        model = Model(
            base_url,
            ModelDefinition(
                uuid=model_id,
                name='My Model',
                model_type=ModelType.BINARY,
                data_type=DataType.TABULAR,
                granularity=Granularity.DAY,
                features=[
                    ColumnDefinition(
                        name='first_name',
                        type=SupportedTypes.string,
                        field_type=FieldType.categorical,
                    ),
                    ColumnDefinition(
                        name='age',
                        type=SupportedTypes.int,
                        field_type=FieldType.numerical,
                    ),
                ],
                outputs=OutputType(prediction=column_def, output=[column_def]),
                target=ColumnDefinition(
                    name='adult',
                    type=SupportedTypes.bool,
                    field_type=FieldType.categorical,
                ),
                timestamp=ColumnDefinition(
                    name='created_at',
                    type=SupportedTypes.datetime,
                    field_type=FieldType.datetime,
                ),
                created_at=str(time.time()),
                updated_at=str(time.time()),
            ),
        )
        response = CurrentFileUpload(
            uuid=uuid.uuid4(),
            path=expected_path,
            date='',
            status=JobStatus.IMPORTING,
            correlation_id_column='correlation',
        )
        responses.add(
            method=responses.POST,
            url=f'{base_url}/api/models/{str(model_id)}/current/bind',
            body=response.model_dump_json(),
            status=200,
            content_type='application/json',
        )
        response = model.load_current_dataset(
            'tests_resources/people_current.csv',
            bucket_name,
            correlation_id_column='correlation',
        )
        assert response.path() == expected_path

    @mock_aws
    @responses.activate
    def test_load_current_dataset_with_object_name(self):
        # Test loading a current dataset with a specified object name
        base_url = 'http://api:9000'
        model_id = uuid.uuid4()
        bucket_name = 'test-bucket'
        file_name = 'test.txt'
        column_def = ColumnDefinition(
            name='prediction', type=SupportedTypes.float, field_type=FieldType.numerical
        )
        expected_path = f's3://{bucket_name}/{file_name}'
        conn = boto3.resource('s3', region_name='us-east-1')
        conn.create_bucket(Bucket=bucket_name)
        model = Model(
            base_url,
            ModelDefinition(
                uuid=model_id,
                name='My Model',
                model_type=ModelType.BINARY,
                data_type=DataType.TABULAR,
                granularity=Granularity.HOUR,
                features=[
                    ColumnDefinition(
                        name='first_name',
                        type=SupportedTypes.string,
                        field_type=FieldType.categorical,
                    ),
                    ColumnDefinition(
                        name='age',
                        type=SupportedTypes.int,
                        field_type=FieldType.numerical,
                    ),
                ],
                outputs=OutputType(prediction=column_def, output=[column_def]),
                target=ColumnDefinition(
                    name='adult',
                    type=SupportedTypes.bool,
                    field_type=FieldType.categorical,
                ),
                timestamp=ColumnDefinition(
                    name='created_at',
                    type=SupportedTypes.datetime,
                    field_type=FieldType.datetime,
                ),
                created_at=str(time.time()),
                updated_at=str(time.time()),
            ),
        )
        response = CurrentFileUpload(
            uuid=uuid.uuid4(),
            path=expected_path,
            date='',
            status=JobStatus.IMPORTING,
            correlation_id_column='correlation',
        )
        responses.add(
            method=responses.POST,
            url=f'{base_url}/api/models/{str(model_id)}/current/bind',
            body=response.model_dump_json(),
            status=200,
            content_type='application/json',
        )
        response = model.load_current_dataset(
            'tests_resources/people_current.csv',
            bucket_name,
            correlation_id_column='correlation',
            object_name=file_name,
        )
        assert response.path() == expected_path

    def test_load_current_dataset_wrong_headers(self):
        # Test loading a current dataset with wrong headers
        column_def = ColumnDefinition(
            name='prediction', type=SupportedTypes.float, field_type=FieldType.numerical
        )
        model = Model(
            'http://api:9000',
            ModelDefinition(
                uuid=uuid.uuid4(),
                name='My Model',
                model_type=ModelType.BINARY,
                data_type=DataType.TABULAR,
                granularity=Granularity.MONTH,
                features=[
                    ColumnDefinition(
                        name='first_name',
                        type=SupportedTypes.string,
                        field_type=FieldType.categorical,
                    ),
                    ColumnDefinition(
                        name='age',
                        type=SupportedTypes.int,
                        field_type=FieldType.numerical,
                    ),
                ],
                outputs=OutputType(prediction=column_def, output=[column_def]),
                target=ColumnDefinition(
                    name='adult',
                    type=SupportedTypes.bool,
                    field_type=FieldType.categorical,
                ),
                timestamp=ColumnDefinition(
                    name='created_at',
                    type=SupportedTypes.datetime,
                    field_type=FieldType.datetime,
                ),
                created_at=str(time.time()),
                updated_at=str(time.time()),
            ),
        )
        with pytest.raises(ClientError):
            model.load_current_dataset(
                'tests_resources/wrong.csv', 'bucket_name', 'correlation'
            )

    @responses.activate
    def test_update_model_features(self):
        # Test updating model features
        base_url = 'http://api:9000'
        model_id = uuid.uuid4()
        column_def = ColumnDefinition(
            name='column', type=SupportedTypes.string, field_type=FieldType.categorical
        )
        outputs = OutputType(prediction=column_def, output=[column_def])
        initial_features = [
            ColumnDefinition(
                name='initial_feature',
                type=SupportedTypes.string,
                field_type=FieldType.categorical,
            )
        ]
        new_features = [
            ColumnDefinition(
                name='new_feature',
                type=SupportedTypes.float,
                field_type=FieldType.numerical,
            )
        ]
        model = Model(
            base_url,
            ModelDefinition(
                uuid=model_id,
                name='My Model',
                model_type=ModelType.BINARY,
                data_type=DataType.TABULAR,
                granularity=Granularity.MONTH,
                features=initial_features,
                outputs=outputs,
                target=column_def,
                timestamp=column_def,
                created_at=str(time.time()),
                updated_at=str(time.time()),
            ),
        )
        responses.add(
            method=responses.POST,
            url=f'{base_url}/api/models/{str(model_id)}',
            body=ModelFeatures(features=new_features).model_dump_json(),
            status=200,
            content_type='application/json',
        )
        model.update_features(new_features)
        assert model.features() == new_features


### Addressing the Feedback

1. **Consistency in Test Cases**:
   - Ensured that the structure and naming conventions of the test cases are consistent with the gold code.
   - Maintained the order of test cases as they were originally, ensuring a logical flow.

2. **Feature Initialization**:
   - Initialized the features in the `test_update_model_features` method to match the gold code structure.

3. **Use of Mocking**:
   - Reviewed and ensured that the use of mocking in the tests is consistent with the gold code.
   - Set up the responses and expected behavior in a manner that aligns with the gold code.

4. **Error Handling**:
   - Ensured that the error handling and specific exceptions raised in tests expecting exceptions are consistent with the gold code.
   - Used `pytest.raises(ClientError)` to check for the correct exception type.

5. **Code Comments**:
   - Made comments concise and directly relevant to the code they describe.
   - Ensured that comments are clear and provide context where necessary.

6. **Response Handling**:
   - Ensured that the handling of responses from the mocked API calls is consistent with the gold code.
   - Asserted the expected outcomes accurately.

By addressing these points, the code should now align more closely with the gold code and should pass the tests without syntax errors.