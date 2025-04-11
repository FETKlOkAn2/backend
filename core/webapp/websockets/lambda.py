import json
import logging
import os
import boto3

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize DynamoDB for connection tracking
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['CONNECTIONS_TABLE'])

# API Gateway Management API client
def get_api_client(domain_name, stage):
    return boto3.client('apigatewaymanagementapi', endpoint_url=f'https://{domain_name}/{stage}')

def lambda_handler(event, context):
    """Handle WebSocket connections and messages"""
    try:
        # Extract connection information
        connection_id = event.get('requestContext', {}).get('connectionId')
        domain_name = event.get('requestContext', {}).get('domainName')
        stage = event.get('requestContext', {}).get('stage')
        
        if not connection_id:
            logger.error("No connection ID found in event")
            return {'statusCode': 400}
        
        # Handle different route types
        route_key = event.get('requestContext', {}).get('routeKey')
        
        if route_key == '$connect':
            # Store connection in DynamoDB
            table.put_item(Item={
                'connectionId': connection_id,
                'timestamp': int(context.time * 1000),
                'user': event.get('queryStringParameters', {}).get('user', 'anonymous')
            })
            return {'statusCode': 200}
            
        elif route_key == '$disconnect':
            # Remove connection from DynamoDB
            table.delete_item(Key={'connectionId': connection_id})
            return {'statusCode': 200}
            
        elif route_key == 'sendMessage':
            # Process message and broadcast to clients
            body = json.loads(event.get('body', '{}'))
            message = body.get('message', '')
            message_type = body.get('type', 'log_update')
            
            # Get API client for sending messages
            api_client = get_api_client(domain_name, stage)
            
            # Get active connections
            connections = table.scan()['Items']
            
            # Broadcast message to all connections
            for connection in connections:
                try:
                    api_client.post_to_connection(
                        ConnectionId=connection['connectionId'],
                        Data=json.dumps({
                            'type': message_type,
                            'message': message
                        })
                    )
                except Exception as e:
                    logger.error(f"Error sending message to {connection['connectionId']}: {str(e)}")
                    
            return {'statusCode': 200}
            
        else:
            logger.error(f"Unknown route key: {route_key}")
            return {'statusCode': 400}
            
    except Exception as e:
        logger.error(f"Error processing WebSocket event: {str(e)}")
        return {'statusCode': 500}