def ALPACA_CONFIG(paper):
    if paper == True:
        ALPACA_CREDS = {
            "API_KEY": "PKYO9RDKK6CLGPA1P6UA",
            "API_SECRET": "McuseZFp8YbMn4TJAi9uSnvAnSkgla1LKVRPv5qk",
            "endpoint": "https://paper-api.alpaca.markets/v2"
        }
        return ALPACA_CREDS
    else:
        ALPACA_CREDS = {
            "API_KEY": "PKYO9RDKK6CLGPA1P6UA",
            "API_SECRET": "McuseZFp8YbMn4TJAi9uSnvAnSkgla1LKVRPv5qk",
            "endpoint": "https://paper-api.alpaca.markets/v2"
        }
        return ALPACA_CREDS