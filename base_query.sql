
--- ## query used

Select cl.EntityId ClaimID,
cl.InvoiceDate InvoiceDate,
cl.ClaimDate ClaimDate,

Ci.EntityClaimEID ClaimItemId,
Ci.ItemClaimedDate ItemClaimedDate,
Ci.ItemDescription ItemDescription,
Ci.ItemQuantity ItemQuantity,
Ci.StatusDate ClaimItemStatusDate,

p.FirstName +' '+p.LastName ClaimantName, 
p.FirstName +' '+p.LastName ClaimantName, 

cl.InvoiceNumber ClaimName ,
cl.InvoiceDate ProductSoldDate, 
ci.ItemClaimedDate ClaimSubmitionDate, 
o.OrgName DealerName,
pe.RefKey ProductID,
pe.Status ProductStatus,

pr.ProductName ProductName, 
pr.HasProductUniqueId IsSerializedProduct, 
pr.ProductName ProductName,
pr.ProductDescription ProductDescription,
pr.ProductManafacturer,

ci.ProductUniqueId SerialNumber,
cs.ListItemName ClaimItemStatus,
cr.ListItemName ClaimItemStatusReason, 
s.EntityId SalesTransactionID, 
s.Quantity SaleQuantity,
s.Amount SaleAmount,
EAC.CommentText AuditComment

from CPEntity.EntityClaim cl
join CPEntity.EntityClaimItem ci on cl.EntityId = ci.EntityClaimEID
Join CPEntity.EntityPersonPosition pp on ci.PrimaryClaimantEID = pp.EntityId
Join CPEntity.EntityPerson p on pp.PersonEID = p.EntityId
Join CPEntity.EntityOrg o on pp.OrgEID = o.EntityId
Join CPEntity.EntityProduct pr on ci.ProductEID = pr.EntityId
Join CPEntity.Entity pe on ci.ProductEID = pe.EntityId
left Join cpdata.ListItem cs on ci.LIClaimItemStatusId =cs.ListItemId
left Join cpdata.ListItem cr on ci.LIClaimItemStatusReasonId =cr.ListItemId
left Join CPEntity.EntityClaimSale cls on ci.EntityId = cls.ClaimItemEID
Left join CPEntity.EntitySale s on cls.SaleEID = s.EntityId
Left join (Select EAC.* from CPEntity.EntityAttribComment EAC -- and licomm.listitemcode='ClaimInternalAuditComment'
 join cpdata.ListItem liComm on liComm.ListItemId = eac.LICommentTypeId and licomm.listitemcode  ='ClaimInternalAuditComment') EAC  on ci.EntityId =eac.EntityId
---128800