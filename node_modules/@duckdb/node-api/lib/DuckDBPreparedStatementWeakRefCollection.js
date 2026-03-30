"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DuckDBPreparedStatementWeakRefCollection = void 0;
class DuckDBPreparedStatementWeakRefCollection {
    preparedStatements = [];
    lastPruneTime = 0;
    add(prepared) {
        const now = performance.now();
        if (now - this.lastPruneTime > 1000) {
            this.lastPruneTime = now;
            this.prune();
        }
        this.preparedStatements.push(new WeakRef(prepared));
    }
    destroySync() {
        for (const preparedRef of this.preparedStatements) {
            const prepared = preparedRef.deref();
            if (prepared) {
                prepared.destroySync();
            }
        }
        this.preparedStatements = [];
    }
    prune() {
        this.preparedStatements = this.preparedStatements.filter((ref) => !!ref.deref());
    }
}
exports.DuckDBPreparedStatementWeakRefCollection = DuckDBPreparedStatementWeakRefCollection;
