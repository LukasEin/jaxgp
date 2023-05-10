import dropbox
from dropbox.exceptions import AuthError, HttpError
import io

import jax.numpy as jnp

DROPBOX_ACCESS_TOKEN = "sl.BeGdRYpTpczw4yHf0zZLnmoIwk5pzbJ1WcJF3IznblssjBH_BNjT6LAZNPJ3M1rqEdO5BXdJqXhgHVlPXypPJ7iyOHx532-TWW5d5YTIfKNLqMLmcb4El102edVujMXHLiZG5vWmO3_C"

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

        out = {}
        for key, value in data.items():
            out[key] = value

        return out
