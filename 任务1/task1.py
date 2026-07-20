import hashlib


def little_endian_to_int(hex_str):
    #将小端序十六进制字符串转换为整数
    return int.from_bytes(bytes.fromhex(hex_str), byteorder='little')


def double_sha256(hex_string):
    #计算双重SHA256
    first_hash = hashlib.sha256(bytes.fromhex(hex_string)).digest()
    second_hash = hashlib.sha256(first_hash).digest()
    return second_hash[::-1].hex()


def parse_block_header(header_hex):

    print("      解析 Block Header       ")
    header_hex = header_hex.strip()

    version = header_hex[0:8]
    prev_hash = header_hex[8:72]
    merkle_root = header_hex[72:136]
    timestamp = header_hex[136:144]
    bits = header_hex[144:152]
    nonce = header_hex[152:160]

    print(f"1. Version (版本号): {version} -> {little_endian_to_int(version)}")
    print(f"2. Previous Block Hash: \n   {prev_hash}")
    print(f"3. Merkle Root: \n   {merkle_root}")
    print(f"4. Timestamp (时间戳): {timestamp} -> {little_endian_to_int(timestamp)}")
    print(f"5. Difficulty Target (Bits): {bits}")
    print(f"6. Nonce (随机数): {nonce} -> {little_endian_to_int(nonce)}")

    calculated_hash = double_sha256(header_hex)
    print(f"\n[PoW 验证] 计算本区块头的双重 SHA256 哈希:\n{calculated_hash}")


def parse_transaction(raw_tx):
    print("      解析 Transaction      ")

    raw_tx = raw_tx.strip()
    cursor = 8
    version = raw_tx[:8]
    print(f"Version: {version} -> {little_endian_to_int(version)}")

    if raw_tx[cursor:cursor + 4] == "0001":
        cursor += 4

    in_count_hex = raw_tx[cursor:cursor + 2]
    in_count = little_endian_to_int(in_count_hex)
    print(f"\nInput Count (输入数量): {in_count_hex} -> {in_count}")
    cursor += 2

    for i in range(in_count):
        print(f"\n--- [Input {i}] ---")
        txid_hex = raw_tx[cursor:cursor + 64]
        print(f"Previous TX Hash: \n  {bytes.fromhex(txid_hex)[::-1].hex()}")
        cursor += 64

        vout = raw_tx[cursor:cursor + 8]
        print(f"Previous Output Index: {vout} -> {little_endian_to_int(vout)}")
        cursor += 8

        script_sig_len = little_endian_to_int(raw_tx[cursor:cursor + 2])
        cursor += 2
        print(f"Unlocking Script Length: {script_sig_len} 字节")

        if script_sig_len > 0:
            script_sig = raw_tx[cursor:cursor + (script_sig_len * 2)]
            print(f"Unlocking Script: {script_sig}")
            cursor += (script_sig_len * 2)
        else:
            print(f"Unlocking Script: 空 ")

        sequence = raw_tx[cursor:cursor + 8]
        print(f"Sequence Number: {sequence}")
        cursor += 8

    out_count_hex = raw_tx[cursor:cursor + 2]
    out_count = little_endian_to_int(out_count_hex)
    print(f"\nOutput Count (输出数量): {out_count_hex} -> {out_count}")
    cursor += 2

    for i in range(out_count):
        print(f"\n--- [Output {i}] ---")
        amount = raw_tx[cursor:cursor + 16]
        satoshi = little_endian_to_int(amount)
        print(f"Amount (聪/Satoshi): {amount} -> {satoshi} ({satoshi / 1e8} tBTC)")
        cursor += 16

        script_pubkey_len = little_endian_to_int(raw_tx[cursor:cursor + 2])
        cursor += 2
        print(f"Locking Script Length: {script_pubkey_len} 字节")

        script_pubkey = raw_tx[cursor:cursor + (script_pubkey_len * 2)]
        print(f"Locking Script: \n  {script_pubkey}")
        cursor += (script_pubkey_len * 2)

    locktime = raw_tx[-8:]
    print(f"\nLocktime (锁定时间): {locktime} -> {little_endian_to_int(locktime)}")


tx_hex = "02000000000101bfa57dc098d76c993f66084b5bcbefd25fd185f65cf07f2fa2edf58dd3cdff5f0000000000fdffffff021027000000000000160014c8c43f9b09e2aadeb3fc1d200da042443bfd3b90a47d0100000000001600148720f7d193ac7dacf3ccee15d44a5a1b970aac290247304402203b2be407b36e2e30da0e4910d59ba4fca11773762a470a3d9726f0ce43470426022016f44f3b79a993556657d31f3bd15e2cad3bb56699e6579e6dc82652b4aecb47012102fbc4a496077b6a3368989f25e72e60b77270b36642689dbca75f731a693f4b1d00000000"

header_hex = "04000020aa40636186cdd62d715c429ca2d7e3b7b09056f700f5f72dec254b3300000000c80a53071babea49096231d076ce91fc8509f15c99b56dc6f7de05e0feb794118adb5d6affff001d59f13a5f"

if __name__ == "__main__":
    parse_block_header(header_hex)
    parse_transaction(tx_hex)