-- An IsCode filter is not a named material that a substitute can replace.
local s,id=GetID()
function s.initial_effect(c)
 c:EnableReviveLimit()
 Fusion.AddProcMix(c,true,true,aux.FilterBoolFunction(Card.IsCode,34460851),44287299)
end
