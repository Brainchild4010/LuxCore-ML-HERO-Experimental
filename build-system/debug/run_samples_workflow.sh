#!/bin/bash

# check whether user had supplied -h or --help . If yes display usage
if [[ ( $@ == "--help") ||  $@ == "-h" ]]
then
	echo "Usage: $0 <repository> <branch>"
	exit 0
fi


gh workflow run "LuxCore Samples Builder" --repo $1 --ref $2
gh run list --workflow="sample-builder.yml" --repo $1
