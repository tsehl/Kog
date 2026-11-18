-- Rule test only: three copies of the same material.
local s,id=GetID()
function s.initial_effect(c)
 c:EnableReviveLimit()
 Fusion.AddProcMixN(c,false,true,89631139,3)
end
