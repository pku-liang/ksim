#ifndef KSIM_ATTRS_TD
#define KSIM_ATTRS_TD

include "ksim/KSimDialect.td"
include "mlir/IR/AttrTypeBase.td"

class KSimAttr<string name, list<Trait> traits=[]> : AttrDef<KSimDialect, name, traits>;

#endif
