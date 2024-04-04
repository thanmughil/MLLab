import re
from urllib.parse import urlparse, parse_qs, unquote

def extract_features(url):
    features = {
        'url_length': len(url),
        'path_length': 0,
        'ip_present': 0,
        'number_of_digits': 0,
        'number_of_parameters': 0,
        'number_of_fragments': 0,
        'is_url_encoded': 0,
        'number_encoded_char': 0,
        'host_length': 0,
        'number_of_periods': 0,
        'has_client': 0,
        'has_admin': 0,
        'has_server': 0,
        'has_login': 0,
        'number_of_dashes': 0,
        'have_at_sign': 0,
        'is_tiny_url': 0,
        'number_of_spaces': 0,
        'number_of_zeros': 0,
        'number_of_uppercase': 0,
        'number_of_lowercase': 0,
        'number_of_double_slashes': 0,
    }
    
    parsed_url = urlparse(url)
    
    features['url_length'] = len(url)
    
    features['path_length'] = len(parsed_url.path)
    
    if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', parsed_url.netloc):
        features['ip_present'] = 1
    
    features['number_of_digits'] = sum(c.isdigit() for c in url)
    
    features['number_of_parameters'] = len(parse_qs(parsed_url.query))
    
    if parsed_url.fragment:
        features['number_of_fragments'] = len(parsed_url.fragment)
    
    if url != unquote(url):
        features['is_url_encoded'] = 1
    
    features['number_encoded_char'] = len(re.findall('%[0-9A-Fa-f]{2}', url))
    
    features['host_length'] = len(parsed_url.netloc)
    
    features['number_of_periods'] = parsed_url.netloc.count('.')
    
    keywords = ['client', 'admin', 'server', 'login']
    for keyword in keywords:
        features['has_' + keyword] = 1 if keyword in url else 0
    
    features['number_of_dashes'] = url.count('-')
    
    features['have_at_sign'] = 1 if '@' in url else 0
    
    features['is_tiny_url'] = 1 if len(url) < 25 else 0
    
    features['number_of_spaces'] = url.count(' ')
    
    features['number_of_zeros'] = url.count('0')
    
    features['number_of_uppercase'] = sum(c.isupper() for c in url)
    
    features['number_of_lowercase'] = sum(c.islower() for c in url)
    
    features['number_of_double_slashes'] = url.count('//')
    
    return features

# url = "https://www.example.com/path/to/page?param1=value1&param2=value2#section1"
# features = extract_features(url)
# for key, value in features.items():
#     print(f"{key}: {value}")
