import dropbox
from dropbox.exceptions import AuthError, HttpError
import io

import jax.numpy as jnp

with open("../dropboxtoken.txt") as f:
    DROPBOX_ACCESS_TOKEN = f.read()

def connect():
    try:
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

    except AuthError as err:
        print(f"Error: {err}")

    return dbx

def db_to_jnp(dbx: dropbox.Dropbox, path): 
    try:
        metadata, result = dbx.files_download(path)
    except HttpError as err:
        print('*** HTTP error', err)
        return None
    
    with io.BytesIO(result.content) as stream:
        data = jnp.load(stream) 

        out = []
        for key, value in data.items():
            out.append(value)
        return out
