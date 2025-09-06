from flask import Response

from routes import app

# Hardcoded payloads as strings
PAYLOADS = {
    "crackme": "111-1111111\n",
    "sqlinject": "Alice'; UPDATE salary SET salary = 999999 WHERE name='Alice'; --\n",
    "stack": "STACK_PAYLOAD_HERE_AS_STRING\n",
    "shellcode": "SHELLCODE_PAYLOAD_AS_STRING\n",
    "hashclash_mini": "HASHCLASH_MINI_PAYLOAD_AS_STRING\n",
    "malicious_mini": "MALICIOUS_MINI_PAYLOAD_AS_STRING\n",
    "hashclash": "HASHCLASH_PAYLOAD_AS_STRING\n",
    "malicious": "MALICIOUS_PAYLOAD_AS_STRING\n"
}

@app.route("/payload_<name>", methods=["GET"])
def get_payload(name):
    payload = PAYLOADS.get(name)
    if payload is None:
        return "Payload not found", 404
    # Return the payload directly as plain text
    return Response(payload, mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
