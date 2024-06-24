#!/bin/bash

# ログファイルへのパス
LOG_FILE_PATH=$1

# ディレクトリのベースパス
BASE_DIRECTORY_PATH=$2

# ログファイルから特定のパターンを含む最初の行を検索し、パターンを抽出
PATTERN=$(grep -oPm1 "(?<=$BASE_DIRECTORY_PATH/)[^/]+" "$LOG_FILE_PATH" | head -1)

# パターンが見つかったかどうかを確認
if [ -z "$PATTERN" ]; then
    echo "エラー: パターンが見つかりませんでした。"
    exit 1
else
    # パターンの最初の8文字を抽出
    echo "${PATTERN:0:8}"
fi
