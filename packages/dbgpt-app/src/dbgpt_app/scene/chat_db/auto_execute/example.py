from dbgpt.core._private.example_base import ExampleSelector, ExampleType

## Two examples are defined by default
EXAMPLES = [
    {
        "messages": [
            {
                "type": "human",
                "data": {
                    "content": "Query the city where user test1 is located",
                    "example": True,
                },
            },
            {
                "type": "ai",
                "data": {
                    "content": """{\n\"thoughts\": \"\
                    Query the user table where user_name is 'test1'\",\
                    \n\"sql\": \"SELECT city FROM user where user_name='test1'\"}""",
                    "example": True,
                },
            },
        ]
    },
    {
        "messages": [
            {
                "type": "human",
                "data": {
                    "content": "Query order information for users in Chengdu",
                    "example": True,
                },
            },
            {
                "type": "ai",
                "data": {
                    "content": """{\n\"thoughts\":\
                     \"Join user and order by user_name; filter city='Chengdu'\",\
                     \n\"sql\": \"SELECT b.* FROM user a  LEFT JOIN tran_order b ON \
                     a.user_name=b.user_name  where a.city='Chengdu'\"}""",
                    "example": True,
                },
            },
        ]
    },
]

sql_data_example = ExampleSelector(
    examples_record=EXAMPLES, use_example=True, type=ExampleType.ONE_SHOT.value
)
