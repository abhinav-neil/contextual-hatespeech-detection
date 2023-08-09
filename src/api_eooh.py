import os
import argparse
from argparse import Namespace
from sys import path
import requests
from dotenv import load_dotenv

def get_paths():
    '''Returns the API paths.'''
    paths = Namespace(
        # login
        login       = '/request_token',

        # channels
        new_channel = '/channel/new',
        my_channels = '/channels/mine',
        get_channel = '/channel/{uid}',

        # lexicons
        add_lexicon    = '/lexicon/add',
        my_lexicons    = '/mylexicons',
        get_lexicon    = '/lexicon/{uid}',
        delete_lexicon = '/lexicon/{uid}/delete'
    )
    return paths

def request_api(method, endpoint, *args, **kwargs):
    '''Wrapper for requests.get and requests.post that adds the auth token.'''
    headers = kwargs.setdefault('headers', {})
    token = login()
    headers['Authorization'] = f'Bearer {token}'
    url = 'https://api.eooh.ai' + endpoint
    response = method(url, *args, **kwargs)

    # check for expired token and refresh if necessary
    if response.status_code == 401 and 'accessToken expired' in response.text:
        print("Access token expired. Refreshing...")
        new_token = login()
        print(f"New token: {new_token}")
        headers['Authorization'] = f'Bearer {new_token}'
        return method(url, *args, **kwargs)  # Retry request with new token

    return response

def login():
    """Returns an authenticating token, valid for 24 hours."""
    # check if token is already set
    load_dotenv()
    token = os.getenv('EOOH_API_KEY')
    # else, login and get new token
    if not token:
        username = os.getenv('EOOH_USERNAME')
        pwd = os.getenv('EOOH_PASSWORD')
        if not username or not pwd:
            username = input("username: ")
            pwd = input("password: ")
        TFA_code = input("2FA code: ")
        payload = {'username': username, 'password': pwd + TFA_code}
        paths = get_paths()
        url = 'https://api.eooh.ai' + paths.login
        response = requests.post(url, data=payload)
        if not response:
            raise ValueError(f'login failed: ({response.status_code}) - {response.text}')
        token = response.json()['accessToken']
        print(f"new token: {token}")

    return token

def create_channel(lexicon_path):
    '''Creates a new channel with the given lexicon.'''
    # read lexicon file
    with open(lexicon_path, 'r', encoding='utf-8') as file:
        lexicon = file.read().splitlines()

    # create a new channel
    channel_payload = {
        'name': 'test channel',
        'public': False,
        'search_parameters': {
            'query': {
                'en': lexicon
            },
            'platform': {
                'Twitter': {},
                'Facebook': {},
                'Reddit': {
                    'path': ['reddit.com/r/politics', 'reddit.com/r/darkjokes', 'reddit.com/r/meanjokes', 'reddit.com/r/offensivejokes']
                }
            }
        }
    }
    paths = get_paths()
    response = request_api(requests.post, paths.new_channel, json=channel_payload)
    channel = response.json()
    print(f"new channel: {channel}")
    
def download_channel(channel_uid):
    '''Downloads an existing channel by its UID.'''
    paths = get_paths()
    response = request_api(requests.get, paths.get_channel.format(uid=channel_uid))
    
    if response.status_code != 200:
        raise ValueError(f'Failed to download channel: ({response.status_code}) - {response.text}')
    
    channel_data = response.json()
    return channel_data


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='EOOH API script.')
#     parser.add_argument('--lexicon-path', type=str, default='queer_lexicon.txt',
#                         help='Path to the lexicon file.')
#     args = parser.parse_args()

#     create_channel(args.lexicon_path)
