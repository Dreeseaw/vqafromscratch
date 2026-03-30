"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DuckDBTimeNSValue = void 0;
exports.timeNSValue = timeNSValue;
const dateTimeStringConversion_1 = require("../conversion/dateTimeStringConversion");
class DuckDBTimeNSValue {
    nanos;
    constructor(nanos) {
        this.nanos = nanos;
    }
    toString() {
        return (0, dateTimeStringConversion_1.getDuckDBTimeStringFromNanosecondsInDay)(this.nanos);
    }
    static Max = new DuckDBTimeNSValue(24n * 60n * 60n * 1000n * 1000n * 1000n);
    static Min = new DuckDBTimeNSValue(0n);
}
exports.DuckDBTimeNSValue = DuckDBTimeNSValue;
function timeNSValue(nanos) {
    return new DuckDBTimeNSValue(nanos);
}
