-- Rule test only: a Warrior and a different Normal Monster.
local s,id=GetID()
function s.initial_effect(c)
 c:EnableReviveLimit()
 Fusion.AddProcMix(c,false,true,aux.FilterBoolFunctionEx(Card.IsRace,RACE_WARRIOR),aux.FilterBoolFunctionEx(Card.IsType,TYPE_NORMAL))
end
