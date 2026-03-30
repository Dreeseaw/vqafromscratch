"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.quotedString = quotedString;
exports.quotedIdentifier = quotedIdentifier;
function quotedString(input) {
    return `'${input.replaceAll(`'`, `''`)}'`;
}
function quotedIdentifier(input) {
    return `"${input.replaceAll(`"`, `""`)}"`;
}
