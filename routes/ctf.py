from flask import Flask, Response, send_file
import os

from routes import app

# Base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Hardcoded payloads: text as strings, binaries as absolute file paths
PAYLOADS = {
    "crackme": "111-1111111\n",
    "sqlinject": "Alice'; UPDATE salary SET salary = 999999 WHERE name='Alice'; --\n",
    "stack": os.path.join(BASE_DIR, "payload_stack"),           # binary file
    "shellcode": os.path.join(BASE_DIR, "payload_shellcode"),   # binary file
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

    # Serve file if payload points to an existing file
    if os.path.exists(payload) and os.path.isfile(payload):
        try:
            return send_file(payload, as_attachment=True)
        except Exception as e:
            return f"Error sending payload: {e}", 500

    # Otherwise return the payload string
    return Response(payload, mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
