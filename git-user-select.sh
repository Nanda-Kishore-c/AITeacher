#!/bin/bash

echo "Select git user:"
echo "1) KP (head.iie@vnrvjiet.in)"
echo "2) NewUser (newuser@example.com)"
echo "3) Custom user"

read -p "Enter choice (1-3): " choice

case $choice in
    1)
        git config user.name "KP"
        git config user.email "head.iie@vnrvjiet.in"
        echo "Set user to: KP (head.iie@vnrvjiet.in)"
        ;;
    2)
        git config user.name "NewUser"
        git config user.email "newuser@example.com"
        echo "Set user to: NewUser (newuser@example.com)"
        ;;
    3)
        read -p "Enter name: " custom_name
        read -p "Enter email: " custom_email
        git config user.name "$custom_name"
        git config user.email "$custom_email"
        echo "Set user to: $custom_name ($custom_email)"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo "Current git user: $(git config user.name) <$(git config user.email)>"
