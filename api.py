import os
import argparse
from argparse import Namespace
import requests
from dotenv import load_dotenv
# from getpass import getpass

API = 'https://api.eooh.ai'
PATHS = Namespace(
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
TOKEN = None

def request_api(method, endpoint, *args, **kwargs):
    '''Wrapper for requests.get and requests.post that adds the auth token.'''
    headers = kwargs.setdefault('headers', {})
    if TOKEN:
        headers['Authorization'] = f'Bearer {TOKEN}'

    url = API+endpoint
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
    username = os.getenv('EOOH_USERNAME')
    pwd = os.getenv('EOOH_PASSWORD')
    if not username or not pwd:
        username = input("username: ")
        pwd = input("password: ")
    TFA_code = input("2FA code: ")
    payload = {'username': username, 'password': pwd + TFA_code}
    response = request_api(requests.post, PATHS.login, data=payload)
    if not response:
        raise ValueError(f'Login Failed: ({response.status_code}) - {response.text}')
    token = response.json()['accessToken']

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

    response = request_api(requests.post, PATHS.new_channel, json=channel_payload)
    channel = response.json()
    print(f"New channel: {channel}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EOOH API script.')
    parser.add_argument('--lexicon-path', type=str, default='queer_lexicon.txt',
                        help='Path to the lexicon file.')
    args = parser.parse_args()

    load_dotenv()
    TOKEN = os.getenv('EOOH_API_KEY')
    if not TOKEN:
        TOKEN = login()
        print(f"New token: {TOKEN}")

    create_channel(args.lexicon_path)
