-- Two cards with the substitute property are allowed if only one of them replaces a name.
local s,id=GetID()
function s.initial_effect(c)
 c:EnableReviveLimit()
 Fusion.AddProcMix(c,true,true,aux.FilterBoolFunctionEx(Card.IsType,TYPE_EFFECT),34460851)
end
