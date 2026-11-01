#!/bin/bash
# container health probe without curl/wget in the image: GET /health over bash's /dev/tcp
exec 3<>"/dev/tcp/127.0.0.1/${LLAMA_ARG_PORT:-8080}" || exit 1
printf 'GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n' >&3
read -r -t 5 _ code _ <&3 || exit 1
[ "$code" = 200 ]
