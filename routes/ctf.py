from flask import Flask, send_file

from routes import app

# === Hardcoded payloads file paths ===
PAYLOADS = {
    "crackme": "111-1111111\n",
    "stack": "Alice'; UPDATE salary SET salary = 999999 WHERE name='Alice'; --\n",
    "shellcode": "./payload_shellcode",
    "hashclash_mini": "./payload_homework_mini",
    "malicious_mini": "./payload_malicious_mini",
    "hashclash": "./payload_homework",
    "malicious": "./payload_malicious",
    "sqlinject": "./payload_sqlinject"  # optional if needed
}

@app.route("/payload_<name>", methods=["GET"])
def get_payload(name):
    payload_file = PAYLOADS.get(name)
    if payload_file:
        try:
            return send_file(payload_file, as_attachment=True)
        except Exception as e:
            return f"Error sending payload: {e}", 500
    else:
        return "Payload not found", 404

if __name__ == "__main__":
    # Run on all interfaces for public access
    app.run(host="0.0.0.0", port=5000, debug=True)
