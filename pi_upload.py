#!/usr/bin/env python3
"""Upload one pet image and print healthy confidence, e.g. 87.5%. Python 3.10+."""
import argparse
import getpass
import json
import os
import re
import secrets
import ssl
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REPLY_BYTES = 256 * 1024


class UploadError(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        # Never forward credentials to a redirect target or downgraded connection.
        return None


def validate_server(value):
    parts = urllib.parse.urlsplit(value)
    if parts.username or parts.password or parts.query or parts.fragment or parts.path not in ('', '/'):
        raise UploadError('Use only the canonical server origin, such as https://pets.onrender.com.')
    if not parts.hostname:
        raise UploadError('A valid server hostname is required.')
    if parts.scheme != 'https' and not (parts.scheme == 'http' and parts.hostname in {'localhost', '127.0.0.1', '::1'}):
        raise UploadError('Remote uploads require HTTPS. HTTP is permitted only on loopback for local tests.')
    try:
        parts.port
    except ValueError:
        raise UploadError('Invalid server port.') from None
    return value.rstrip('/')


def read_token(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise UploadError('The token file must be a regular file, not a symlink.')
    if os.name != 'nt' and (info.st_mode & 0o077 or info.st_uid != os.getuid()):
        raise UploadError('The token file must belong to this user and have mode 600. Run chmod 600 on it.')
    with path.open() as handle:
        token = handle.read(129).strip()
    if not re.fullmatch(r'pet_[A-Za-z0-9_-]{43}', token):
        raise UploadError('Invalid device credential in the token file.')
    return token


def api_request(server, path, *, token=None, body=None, content_type=None):
    server = validate_server(server)
    headers = {'Accept': 'text/plain, application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    if content_type:
        headers['Content-Type'] = content_type
    request = urllib.request.Request(server + path, data=body, headers=headers)
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    try:
        with opener.open(request, timeout=120) as response:
            data = response.read(MAX_REPLY_BYTES + 1)
            if len(data) > MAX_REPLY_BYTES:
                raise UploadError('The server response exceeded the allowed size.')
            return data.decode('utf-8')
    except urllib.error.HTTPError as error:
        if error.code in {301, 302, 303, 307, 308}:
            raise UploadError('The server redirected the request. Use its canonical HTTPS origin; redirects are not followed.') from None
        # Hosted platform error pages are not suitable terminal output.
        detail = error.read(2048).decode('utf-8', errors='replace') if error.headers.get_content_type() == 'text/plain' else ''
        detail = ' '.join(detail.split())
        raise UploadError(f'HTTP {error.code}: {detail or "request failed"}') from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise UploadError('Network or TLS failure. Check the server URL, internet connection, system clock, and CA certificates. No automatic retry was made.') from error


def upload_image(server, token, image_path, pet_id=None):
    # Read at most the allowed amount, even if the file grows during reading.
    with image_path.open('rb') as image_file:
        image = image_file.read(MAX_IMAGE_BYTES + 1)
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise UploadError('Choose a non-empty image of at most 10 MB.')
    boundary = 'pet-upload-' + secrets.token_hex(24)
    chunks = []
    if pet_id is not None:
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="pet_id"\r\n\r\n{pet_id}\r\n'.encode())
    # A constant filename prevents arbitrary paths or header characters from being sent.
    chunks.extend([
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="image.jpg"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode(),
        image,
        f'\r\n--{boundary}--\r\n'.encode(),
    ])
    answer = api_request(server, '/api/images', token=token, body=b''.join(chunks),
                         content_type=f'multipart/form-data; boundary={boundary}').strip()
    if not re.fullmatch(r'(?:100(?:\.0+)?|(?:[0-9]|[1-9][0-9])(?:\.[0-9]+)?)%', answer):
        raise UploadError('The server did not return a healthy-confidence percentage. Check that the updated server is deployed. No automatic retry was made.')
    return answer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, nargs='?', help='Existing JPEG, PNG, WebP, or non-animated GIF')
    parser.add_argument('--capture', action='store_true', help='Capture a temporary photo with rpicam-still')
    parser.add_argument('--server', default=os.getenv('PET_SERVER_URL'), help='Canonical HTTPS site origin')
    parser.add_argument('--token-file', type=Path, default=Path.home() / '.config/pet-image/device-token')
    parser.add_argument('--pet-id', type=int, help='Optional additional check against the credential-bound pet')
    parser.add_argument('--username', help='Interactive login instead of a stored device credential')
    parser.add_argument('--pet-name', help='Pet name for interactive login; alternatively use --pet-id')
    args = parser.parse_args(argv)
    if not args.server or bool(args.image) == bool(args.capture):
        parser.error('Supply --server and either an image path or --capture.')
    if args.pet_id is not None and args.pet_id <= 0:
        parser.error('--pet-id must be a positive integer.')
    if args.username and args.pet_id is None and not args.pet_name:
        parser.error('Interactive login requires --pet-id or --pet-name.')
    server, token, temporary_login = None, None, False
    try:
        server = validate_server(args.server)
        if args.username:
            credentials = {'username': args.username, 'password': getpass.getpass('Account password: ')}
            credentials['pet_id' if args.pet_id is not None else 'pet_name'] = args.pet_id if args.pet_id is not None else args.pet_name
            result = json.loads(api_request(server, '/api/login', body=json.dumps(credentials).encode(), content_type='application/json'))
            credentials.clear()
            if not isinstance(result.get('access_token'), str) or not re.fullmatch(r'pet_[A-Za-z0-9_-]{43}', result['access_token']):
                raise UploadError('The server returned an invalid access credential.')
            token, temporary_login = result['access_token'], True
        else:
            token = read_token(args.token_file.expanduser())
        if args.capture:
            with tempfile.TemporaryDirectory(prefix='pet-photo-') as directory:
                image_path = Path(directory) / 'image.jpg'
                subprocess.run(['rpicam-still', '--nopreview', '--timeout', '1000', '--width', '1920', '--height', '1080', '--output', str(image_path)],
                               check=True, timeout=30, stdout=subprocess.DEVNULL)
                answer = upload_image(server, token, image_path, args.pet_id)
        else:
            answer = upload_image(server, token, args.image, args.pet_id)
        print(answer)
        return 0
    except (UploadError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'Upload failed: {error}', file=sys.stderr)
        return 1
    finally:
        if temporary_login:
            try:
                api_request(server, '/api/logout', token=token, body=b'')
            except (UploadError, OSError):
                print('Temporary credential could not be revoked; it expires within one hour.', file=sys.stderr)


if __name__ == '__main__':
    raise SystemExit(main())
