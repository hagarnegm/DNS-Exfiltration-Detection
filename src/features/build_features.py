from collections import Counter

# import enchant
import numpy as np
from nltk import everygrams


def FQDN_count(query):
    """
    :param query: Raw DNS query
    :return: Count of characters in the Fully Qualified Domain Name
    """
    FQDN = "".join(query.split("."))
    return len(FQDN)


def subdomain_length(query):
    """
    :param query: Raw DNS query
    :return: Count of characters in all the subdomains
    """
    subdomain = "".join(query.split(".")[:-2])
    return len(subdomain)


def upper(query):
    """
    :param query: Raw DNS query
    :return: Count of uppercase characters in the domain name
    """
    FQDN = "".join(query.split("."))
    return sum(map(str.isupper, FQDN))


def lower(query):
    """
    :param query: Raw DNS query
    :return: Count of lowercase characters in the domain name
    """
    FQDN = "".join(query.split("."))
    return sum(map(str.islower, FQDN))


def numeric(query):
    """
    :param query: Raw DNS query
    :return: Count of numbers in the domain name
    """
    FQDN = "".join(query.split("."))
    return sum(map(str.isnumeric, FQDN))


def entropy(query):
    """
    :param query: Raw DNS query
    :return: The total randomness/probability of the occurrence of each character in the domain name
    """
    FQDN = "".join(query.split("."))
    counter = Counter(FQDN)
    frequencies = np.array(list(counter.values()))
    p = frequencies / len(FQDN)
    log2p = np.log2(p)
    return -np.sum(np.multiply(p, log2p))


def special(query):
    """
    :param query: Raw DNS query
    :return: Count of special characters in the domain name
    """
    FQDN = "".join(query.split("."))
    numbers = sum(map(str.isnumeric, FQDN))
    letters = sum(map(str.isalpha, FQDN))
    return len(FQDN) - (numbers + letters)


def labels(query):
    """
    :param query: Raw DNS query
    :return: The number of labels in the domain name
    """
    return len(query.split("."))


def labels_max(query):
    """
    :param query: Raw DNS query
    :return: The maximum label length in the domain name
        """
    FQDN = query.split(".")
    return max([len(label) for label in FQDN])


def labels_average(query):
    """
    :param query: Raw DNS query
    :return: The average label length in the domain name
    """
    FQDN = query.split(".")
    return sum([len(label) for label in FQDN]) / len(FQDN)


def longest_word(query):
    """
    :param query: Raw DNS query
    :return: The longest meaningful substring along all the subdomains
    """
    # dct = enchant.Dict("en-US")
    subdomains = query.split(".")[:-2]

    if len(subdomains) == 0:
        return ""

    all_substrings = []
    for label in subdomains:
        if not all(map(str.isnumeric, label)):
            substrings = [[''.join(ngram), len(''.join(ngram))] for ngram in everygrams(label) if #dct.check(''.join(ngram)) and
                          all(map(str.isalpha, ngram))]
            all_substrings.extend(substrings)

    if len(all_substrings) == 0:
        return ""

    return max(all_substrings, key=lambda x: x[1])[0]


def sld(query):
    """
    :param query: Raw DNS query
    :return: The second level domain name
    """
    FQDN = query.split(".")
    if len(FQDN) > 1:
        return FQDN[-2]
    else:
        return ""
    

def length(query):
    """
    :param query: Raw DNS query
    :return: The total length of the subdomains and the second level domain name
    """
    FQDN = query.split(".")
    return len("".join(FQDN[:-1]))


def subdomain(query):
    """
    :param query: Raw DNS query
    :return: A boolean representing whether the full domain name contains subdomains or not
    """
    FQDN = query.split(".")
    return int(len(FQDN) > 2)


def extract_features(query):
    """
    :param query: Raw DNS query
    :return: An array of all the features extracted from the DNS query
    """
    features = [FQDN_count(query), subdomain_length(query), upper(query), lower(query), numeric(query), int(entropy(query)),
                special(query), labels(query), labels_max(query), labels_average(query), longest_word(query),
                sld(query), length(query), subdomain(query)]

    return features
