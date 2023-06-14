import requests
from argparse import Namespace
from getpass import getpass

# Auto-generated documentation: 

API = 'https://api.eooh.ai'
PATHS = Namespace(
    # Login
    login       = '/request_token',

    # Channels
    new_channel = '/channel/new',
    my_channels = '/channels/mine',
    get_channel = '/channel/{uid}',

    # Lexicons
    add_lexicon    = '/lexicon/add',
    my_lexicons    = '/mylexicons',
    get_lexicon    = '/lexicon/{uid}',
    delete_lexicon = '/lexicon/{uid}/delete', 
)
TOKEN = None

def _req(method, endpoint, *args, **kwargs):
    if endpoint != PATHS.login:
        assert TOKEN, 'You must log in first.'
        kwargs.setdefault('headers', {})
        kwargs['headers']['Authorization'] = f'Bearer {TOKEN}'
    
    url = API+endpoint
    return method(url, *args, **kwargs)

get  = lambda *a, **kw: _req(requests.get,  *a, **kw)
post = lambda *a, **kw: _req(requests.post, *a, **kw)

def login(email, password=None, digits=None, refresh=False):
    """Returns an authentifying token, valid for 24 hours."""
    if not TOKEN or refresh:
        payload = {'username':email}
        payload['password'] = password or getpass("Password: ") + (digits or input("2FA digits: "))
        response =  post(PATHS.login, data=payload)
        if not response:
            raise ValueError(f'Login Failed: ({response.status_code}) - {response.text}')
        
        token = response.json()['accessToken']
    return token


if __name__ == '__main__':
    email = 'abhinav.bhuyan@student.uva.nl' 
    TOKEN = login(email)
    
    # Create a new channel
    payload = {
        'name': 'test channel',
        'public':False,
        'search_parameters':{
            'query':{
                'en':['1488', 'oy vey', 'a future for white children']
            },
            'platform':{
                'Twitter':{} # No mandatory fields for Twitter
            }
        }
    }

    response = post(PATHS.new_channel, json=payload)
    channel = response.json()
    print(channel)





















    







