#!/usr/bin/env python3
"""Bounded loopback-only TCP relay; never inspect or log proxy traffic."""
import argparse
import select
import socket
import threading


def relay(client, upstream_port):
    with client, socket.create_connection(('127.0.0.1', upstream_port), timeout=10) as upstream:
        client.settimeout(15)
        upstream.settimeout(15)
        readers = [client, upstream]
        while readers:
            ready, _, _ = select.select(readers, [], [], 300)
            if not ready:
                return
            for source in ready:
                destination = upstream if source is client else client
                data = source.recv(65536)
                if data:
                    destination.sendall(data)
                else:
                    readers.remove(source)
                    destination.shutdown(socket.SHUT_WR)


def serve(listen_port, upstream_port):
    if not (1 <= listen_port <= 65535 and 1 <= upstream_port <= 65535) or listen_port == upstream_port:
        raise ValueError('Distinct valid loopback ports required')
    capacity = threading.BoundedSemaphore(16)

    def handle(client):
        try:
            relay(client, upstream_port)
        except (OSError, TimeoutError):
            client.close()
        finally:
            capacity.release()

    with socket.socket() as listener:
        listener.bind(('127.0.0.1', listen_port))
        listener.listen(16)
        print(f'loopback relay ready: {listen_port} -> {upstream_port}', flush=True)
        while True:
            client, _ = listener.accept()
            if not capacity.acquire(blocking=False):
                client.close()
                continue
            threading.Thread(target=handle, args=(client,), daemon=True).start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen-port', type=int, default=17890)
    parser.add_argument('--upstream-port', type=int, default=17899)
    args = parser.parse_args()
    serve(args.listen_port, args.upstream_port)
