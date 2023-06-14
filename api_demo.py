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
    email = 'pierre@textgain.com' 
    TOKEN = login(email)
    

    # Create a new channel
    payload = {
        'name': 'Far-right dog whistles',
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

    # response = post(PATHS.new_channel, json=payload)
    # channel = response.json()
    # print(channel)

    # List active channels
    my_channels = get(PATHS.my_channels).json()
    ch1 = my_channels[0]

    # Get a specific channel
    print(get(PATHS.get_channel.format(uid=ch1['uid'])).json())


    # Update a custom lexicon
    import base64
    # filepath = '' # Path to your CSV file (first row must be the headers)
    with open(filepath, 'rb') as f:
        raw = f.read()
    encoded = base64.b64encode(raw)
    payload = dict(
        file = encoded.decode('utf8'),
        lang = 'en', # ISO-code
        name = 'My awesome lexicon',
        active_fields = dict( # Fields not in 'active_fields' will be saved but ignored during analysis.
            phrase = 'word',    # Name of the column containing the word or phrase
            score = 'score',    # Name of the column containing the annotation score
            categories = [      # Name of the columns containing categories to be considered. 
                'Racism', 'Misogyny', 'Far-right'
            ]
        )
    )

    response = post(PATHS.add_lexicon, json=payload)
    print(response.json())


    # List existing lexicons
    response = get(PATHS.my_lexicons)
    lexicons = response.json()
    print(lexicons)

    # Get a specific lexicon
    lex = lexicons[0]
    response = get(PATHS.get_lexicon.format(lex['uid']))
    print(response.json())

    # # Delete a specific lexicon
    # lex = lexicons[0]
    # response = get(PATHS.delete_lexicon.format(lex['uid']))
    # print(response.json())



















    







